#!/usr/bin/env python3
"""REVISE multi-round plug-and-play evaluation for Video-MME / LVBench (vLLM).

Runs the REVISE question-aware sparse-video loop against a vLLM
OpenAI-compatible server for the two long-video multiple-choice benchmarks
Video-MME and LVBench (selected via CLI flags). Each round the model reasons in
``<think>``, emits a ``<summarize>`` P/O/H/U/R state, then either ``<select>``s
unseen frames or commits an ``<answer>``; ``_bare_answer_after_summary`` also
recovers a trailing bare letter when no frame request follows the summary.

Frame indexing is 1-fps timeline based (these are long videos). Run as a CLI
(see ``main``); invoked by ``scripts/paper_suite.py`` and imported for cache
coverage checks by ``scripts/check_video_cache_coverage.py`` and
``scripts/cache_hf_video_benchmark.py``. Shared helpers: ``revise/pnp_utils.py``.

NOTE: this file's ``MCVideoSample`` / loaders intentionally differ from the
HF-backend variant in ``plug_and_play_lvbench_hf.py`` (this one keeps every row
and records a ``video_url``; the HF one filters empty answers and carries
``question_type``/``video_type``). They are deliberately not unified -- see
``tests/test_pnp_characterization.py::LoaderDivergenceTest``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

# Allow direct execution via `python examples/...py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise.pnp_prompts import SYSTEM_PROMPT_WITH_THINK as DEFAULT_SYSTEM_PROMPT_WITH_THINK
from revise.pnp_utils import FORCE_ANSWER_INSTRUCTIONS_SIMPLE as _FORCE_ANSWER_INSTRUCTIONS
from revise.pnp_utils import chat_once as _shared_chat_once
from revise.pnp_utils import start_vllm_server as _shared_start_vllm_server
from revise.pnp_utils import (
    ANSWER_RE,
    SELECT_RE,
    SUMMARIZE_RE,
    THINK_RE,
    collapse_ws,
    dedupe_preserve_order,
    ensure_writable_hf_cache,
    extract_frames_1fps,
    extract_tag,
    extract_video_info,
    format_question_block,
    format_videomme_question_block,
    get_api_headers,
    get_model_id,
    maybe_init_wandb,
    maybe_log_jsonl,
    normalize_answer_letter,
    parse_int_list,
    parse_options_from_lvbench_question,
    parse_time_reference_range,
    pick_free_port,
    propose_candidate_frames,
    resolve_base_url,
    sample_uniform_indices_inclusive,
    shard_by_video,
    stable_sample_id_dataset,
    retry_feedback_text,
    stop_server,
    timeline_len_1fps,
    wait_for_server,
    wandb_log,
)

FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are a multiple-choice video QA assistant. "
    "Return exactly <think>...</think> then <answer>LETTER</answer>. "
    "LETTER must be one of the option letters in the question. "
    "Do not output <select>, requests, or any text outside these tags."
)


def _bare_answer_after_summary(raw_output: str) -> Optional[str]:
    """Recover a bare final letter only when no frame-request tag is present."""
    if not raw_output or extract_tag(raw_output, SELECT_RE) is not None:
        return None
    tail = raw_output
    m_end = None
    for m in re.finditer(r"</summarize>", raw_output, flags=re.IGNORECASE):
        m_end = m.end()
    if m_end is not None:
        tail = raw_output[m_end:]
    tail = collapse_ws(tail)
    if not tail or len(tail) > 32:
        return None
    toks = tail.split()
    return toks[-1] if toks else tail


def _retry_feedback_text(feedback: str, *, force_answer: bool) -> str:
    return retry_feedback_text(
        feedback,
        force_answer=force_answer,
        force_instructions=_FORCE_ANSWER_INSTRUCTIONS,
    )


def _start_vllm_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    # Video-MME/LVBench default to a single GPU (cuda_visible_default="0").
    return _shared_start_vllm_server(
        args, image_limit=int(args.max_frames_per_round), cuda_visible_default="0"
    )


def _chat_once(
    base_url: str,
    model_id: str,
    system_prompt: str,
    user_text: str,
    images: list[Image.Image],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    # Long-video frames are downscaled (max_edge=384, q=85) to keep the
    # per-prompt payload small; the shared chat_once takes these as kwargs.
    return _shared_chat_once(
        base_url,
        model_id,
        system_prompt,
        user_text,
        images,
        temperature,
        top_p,
        max_tokens,
        timeout_s,
        max_edge=384,
        quality=85,
    )


def _ensure_yt_dlp(py_bin: str) -> list[str]:
    """Return command prefix to invoke yt-dlp (binary or `python -m yt_dlp`)."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    # Fall back to module invocation.
    return [py_bin, "-m", "yt_dlp"]


def _download_youtube(url: str, out_mp4: str, *, py_bin: str, timeout_s: int) -> None:
    out_mp4_path = Path(out_mp4)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_mp4_path.with_suffix("")) + ".%(ext)s"

    # yt-dlp increasingly requires a JS runtime for YouTube extraction. We prefer Node if present.
    node_path = shutil.which("node")
    js_runtime_args: list[str] = []
    if node_path:
        js_runtime_args = ["--js-runtimes", f"node:{node_path}"]

    cmd = [
        *_ensure_yt_dlp(py_bin),
        *js_runtime_args,
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "--extractor-args",
        "youtube:player_client=android",
        "-f",
        "best[ext=mp4][height<=480]/best[ext=mp4]/best",
        "-o",
        out_tmpl,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed ({proc.returncode}): {proc.stderr.strip()[:500]}")

    # Find the merged mp4.
    if out_mp4_path.exists() and out_mp4_path.stat().st_size > 0:
        return
    # Sometimes yt-dlp still leaves a different extension; try to locate by stem.
    candidates = list(out_mp4_path.parent.glob(out_mp4_path.stem + ".*"))
    for c in candidates:
        if c.suffix.lower() == ".mp4" and c.stat().st_size > 0:
            c.rename(out_mp4_path)
            return
    raise FileNotFoundError(f"Downloaded file not found for {url} (expected {out_mp4_path})")


@dataclass
class MCVideoSample:
    dataset: str
    uid: str
    video_key: str
    video_url: str
    question: str
    options: list[str]
    answer_letter: str
    time_reference: str = ""

    @property
    def sample_id(self) -> str:
        return stable_sample_id_dataset(self.dataset, self.video_key, self.uid)


def load_dataset(*args, **kwargs):
    """Lazily import HF ``datasets`` so this vLLM-client module stays import-light.

    ``datasets`` / ``huggingface_hub`` read their cache-location env vars at
    import time, so point them at the writable asset cache *before* importing.
    Keeping this lazy means tooling that merely imports this module (tests,
    ``--help``, the paper-suite command builder) does not pull in the
    multi-gigabyte HF/torch stack.

    Kept as a module-level name (not ``_load_dataset``) so the loaders' call
    sites remain monkeypatchable as ``<module>.load_dataset`` -- a seam the
    characterization tests rely on.
    """
    ensure_writable_hf_cache(REPO_ROOT / "data" / "revise_assets" / "hf_home")
    from datasets import load_dataset as _hf_load_dataset

    return _hf_load_dataset(*args, **kwargs)


def _load_videomme_samples(split: str) -> list[MCVideoSample]:
    ds = load_dataset("lmms-lab/Video-MME", split=split)
    samples: list[MCVideoSample] = []
    for ex in ds:
        video_id = str(ex.get("videoID") or ex.get("video_id") or "").strip()
        url = str(ex.get("url") or "").strip()
        qid = str(ex.get("question_id") or ex.get("qid") or "").strip()
        question = str(ex.get("question") or "").strip()
        options_raw = ex.get("options") or []
        if not isinstance(options_raw, list):
            options_raw = []
        options: list[str] = []
        for opt in options_raw:
            s = str(opt).strip()
            m = re.match(r"^[A-Z]\s*[.)]\s*(.*)$", s)
            options.append(m.group(1).strip() if m else s)
        answer = str(ex.get("answer") or "").strip().upper()
        samples.append(
            MCVideoSample(
                dataset="videomme",
                uid=qid or stable_sample_id_dataset("videomme", video_id, question),
                video_key=f"{video_id}.mp4",
                video_url=url,
                question=question,
                options=options,
                answer_letter=answer,
            )
        )
    return samples


def _load_lvbench_samples(split: str) -> list[MCVideoSample]:
    ds = load_dataset("lmms-lab/LVBench", split=split)
    samples: list[MCVideoSample] = []
    for ex in ds:
        video_path = str(ex.get("video_path") or "").strip()
        uid = str(ex.get("uid") or ex.get("key") or "").strip()
        q_raw = str(ex.get("question") or "").strip()
        q_text, options = parse_options_from_lvbench_question(q_raw)
        answer = str(ex.get("answer") or "").strip().upper()
        time_reference = str(ex.get("time_reference") or "").strip()
        video_id = Path(video_path).stem
        url = f"https://www.youtube.com/watch?v={video_id}"
        samples.append(
            MCVideoSample(
                dataset="lvbench",
                uid=uid or stable_sample_id_dataset("lvbench", video_path, q_raw),
                video_key=video_path,
                video_url=url,
                question=q_text if q_text else q_raw,
                options=options,
                answer_letter=answer,
                time_reference=time_reference,
            )
        )
    return samples


def _build_user_text(
    question_block: str,
    summary: str,
    timeline_len: int,
    round_idx: int,
    current_frames: list[int],
    seen_frames: list[int],
    candidate_unseen_frames: list[int],
    use_candidate_frame_ids: bool,
    require_candidate_frames: bool,
    time_reference: str = "",
    num_options: int = 0,
) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def _idx_to_letters(idx: int) -> str:
        # Excel-style column labels: 0->A, 25->Z, 26->AA, ...
        if idx < 0:
            return "?"
        base = len(letters)
        n = idx + 1
        out = ""
        while n > 0:
            n -= 1
            n, rem = divmod(n, base)
            out = letters[rem] + out
        return out

    n_opts = num_options if num_options > 0 else max(1, question_block.count("\n") - 1)
    allowed_letters = ", ".join(list(letters[: n_opts]))

    lines: list[str] = []
    lines.append(f"Round {round_idx} / Question:")
    lines.append(question_block)
    if allowed_letters:
        lines.append(
            f"To answer, output <think>...</think> then <answer>LETTER</answer> "
            f"(LETTER must be one of: {allowed_letters})."
        )
    lines.append(f"Total frames L = {timeline_len} (1 fps timeline).")
    if time_reference:
        lines.append(f"Relevant time window for this question: {time_reference} (focus on this segment).")
    lines.append(
        f"Seen frames: {len(seen_frames)} frames already viewed (do NOT request any previously shown frames)."
    )
    if use_candidate_frame_ids and candidate_unseen_frames:
        lines.append(
            f"Candidate unseen frames available as IDs (all NEW): choose IDs in [1, {len(candidate_unseen_frames)}]."
        )
        id_map = ", ".join(f"{i+1}->{t}s" for i, t in enumerate(candidate_unseen_frames))
        lines.append(f"Candidate ID -> timeline second: {id_map}")
        lines.append("In <select>, output ONLY candidate IDs (comma-separated). Do NOT output raw indices when IDs exist.")
        if require_candidate_frames:
            lines.append("IMPORTANT: You MUST choose frames only from the Candidate IDs.")
    lines.extend(["Current summary:", f"<summarize>{summary}</summarize>", "Frames shown in this round:"])
    for i in range(len(current_frames)):
        lines.append(f"Shown frame {_idx_to_letters(i)} <image>")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["videomme", "lvbench"], required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--video-cache-dir", default="./data/revise_assets/video_cache",
                    help="Local cache for downloaded benchmark videos (set REVISE_VIDEO_CACHE_DIR or pass to override)")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=0)

    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Defaults to http://host:port.")
    ap.add_argument("--model-id", default=None, help="Explicit remote model ID for chat completions.")
    ap.add_argument("--start-server", action="store_true")
    ap.add_argument("--restart-server-on-failure", action="store_true")
    ap.add_argument("--server-log", default="")
    ap.add_argument("--server-timeout-s", type=int, default=240)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)

    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-frames-per-round", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--use-candidate-frame-ids", action="store_true", default=True)
    ap.add_argument("--require-candidate-frames", action="store_true", default=True)
    ap.add_argument("--max-retries-per-round", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--force-final-answer", action="store_true", default=True)

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)

    ap.add_argument("--log-jsonl", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--resume-from-log", action="store_true")

    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "verl-revise"))
    ap.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    ap.add_argument("--wandb-name", default=os.getenv("WANDB_RUN_NAME"))
    ap.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", ""))
    ap.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", ""))

    ap.add_argument("--yt-dlp-timeout-s", type=int, default=600)
    ap.add_argument(
        "--videomme-use-official-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the official Video-MME no-subtitle multiple-choice prompt template.",
    )
    ap.add_argument(
        "--cached-only",
        action="store_true",
        help="Only evaluate samples whose videos already exist in --video-cache-dir (skip downloads).",
    )
    ap.add_argument(
        "--allow-missing-cached-videos",
        action="store_true",
        help="Allow full cached-only runs to evaluate a cached subset when videos are missing.",
    )

    args = ap.parse_args()
    if args.base_url and args.start_server:
        raise ValueError("--base-url cannot be combined with --start-server.")

    if args.port <= 0:
        args.port = pick_free_port()

    split = args.split
    if not split:
        split = "test" if args.dataset == "videomme" else "train"

    if args.dataset == "videomme":
        samples = _load_videomme_samples(split)
    else:
        samples = _load_lvbench_samples(split)

    cache_filter_total = len(samples)
    cache_filter_missing = 0
    cache_filter_missing_examples: list[str] = []
    if args.cached_only:
        cache_dir = Path(args.video_cache_dir) / args.dataset
        filtered: list[MCVideoSample] = []
        missing_keys: list[str] = []
        for s in samples:
            video_path = cache_dir / s.video_key
            try:
                ok = video_path.stat().st_size > 0
            except FileNotFoundError:
                ok = False
            if ok:
                filtered.append(s)
            else:
                missing_keys.append(s.video_key)
        unique_missing = set(missing_keys)
        cache_filter_missing = len(unique_missing)
        cache_filter_missing_examples = sorted(unique_missing)[:20]
        if cache_filter_missing and args.max_samples <= 0 and not args.allow_missing_cached_videos:
            total_unique = len({s.video_key for s in samples})
            raise SystemExit(
                f"--cached-only missing {cache_filter_missing} distinct {args.dataset} videos "
                f"out of {total_unique}; examples={cache_filter_missing_examples}. "
                "Populate the cache or pass --allow-missing-cached-videos for an explicitly partial run."
            )
        samples = filtered

    start_idx = max(0, int(args.start_idx or 0))
    end_idx = int(args.end_idx or 0)
    if end_idx <= 0:
        end_idx = len(samples)
    if start_idx > 0 or end_idx < len(samples):
        samples = samples[start_idx:end_idx]

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    samples = shard_by_video(samples, args.num_shards, args.shard_idx)

    # Stable order to improve download locality.
    samples.sort(key=lambda s: (s.video_key, s.uid))

    if not samples:
        raise SystemExit("No samples selected (check --split/--max-samples/--sharding).")

    # Auto-suffix output paths for multi-shard runs.
    if args.num_shards > 1:
        suffix = f".shard{args.shard_idx}of{args.num_shards}"

        def _suffix_path(path: str) -> str:
            root, ext = os.path.splitext(path)
            return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"

        if args.log_jsonl and suffix not in args.log_jsonl:
            args.log_jsonl = _suffix_path(args.log_jsonl)
        if args.summary_json and suffix not in args.summary_json:
            args.summary_json = _suffix_path(args.summary_json)

    resume_completed = 0
    correct = 0
    total_rounds = 0
    invalid_outputs = 0
    invalid_action_terminated = 0
    failed = 0
    total_model_calls = 0
    total_retries = 0
    total_effective_rounds = 0
    think_present = 0
    missing_summary = 0
    answered = 0
    total_frames_used_all = 0
    total_frames_used_answered = 0

    if args.resume_from_log and args.log_jsonl and os.path.exists(args.log_jsonl):
        # Cheap resume heuristic: count answered samples in log.
        seen_samples: set[str] = set()
        with open(args.log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("sample_id")
                if sid and "<answer>" in str(obj.get("raw_output", "")).lower():
                    seen_samples.add(sid)
        resume_completed = len(seen_samples)
        if resume_completed > 0:
            print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}")

    if resume_completed > 0:
        samples = samples[resume_completed:]

    server_proc: Optional[subprocess.Popen[str]] = None
    if args.start_server:
        server_proc = _start_vllm_server(args)
        wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)

    run_config = {
        "task": "revise_plug_and_play_videomme_lvbench_vllm",
        "dataset": args.dataset,
        "split": split,
        "model_path": args.model_path,
        "video_cache_dir": args.video_cache_dir,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_samples": args.max_samples,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "max_rounds": args.max_rounds,
        "max_frames_per_round": args.max_frames_per_round,
        "candidate_k": args.candidate_k,
        "max_retries_per_round": args.max_retries_per_round,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "use_candidate_frame_ids": bool(args.use_candidate_frame_ids),
        "require_candidate_frames": bool(args.require_candidate_frames),
        "force_final_answer": bool(args.force_final_answer),
        "videomme_use_official_prompt": bool(args.videomme_use_official_prompt),
        "cached_only": bool(args.cached_only),
        "allow_missing_cached_videos": bool(args.allow_missing_cached_videos),
        "cache_filter_total_samples": cache_filter_total,
        "cache_filter_missing_videos": cache_filter_missing,
        "cache_filter_missing_examples": cache_filter_missing_examples,
        "log_jsonl": args.log_jsonl,
        "summary_json": args.summary_json,
    }
    run = maybe_init_wandb(args, run_config)

    base_url = resolve_base_url(args.base_url, args.host, args.port)
    model_id = get_model_id(base_url, model_id=args.model_id)
    system_prompt = DEFAULT_SYSTEM_PROMPT_WITH_THINK.format(max_frames_per_round=args.max_frames_per_round)

    rng = random.Random(42 + int(args.shard_idx))

    start_t = time.time()

    def _process_one(sample: MCVideoSample) -> None:
        nonlocal correct, total_rounds, invalid_outputs, invalid_action_terminated, failed
        nonlocal total_model_calls, total_retries, total_effective_rounds, think_present
        nonlocal missing_summary, answered, total_frames_used_all, total_frames_used_answered
        nonlocal server_proc, model_id

        cache_dir = Path(args.video_cache_dir) / sample.dataset
        cache_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(cache_dir / sample.video_key)
        failed_marker = video_path + ".failed"

        video_ok = os.path.exists(video_path) and os.path.getsize(video_path) > 0
        if video_ok and os.path.exists(failed_marker):
            try:
                os.remove(failed_marker)
            except Exception:
                pass

        # Negative-cache download failures (e.g., private/unavailable YouTube videos) to avoid repeated yt-dlp calls
        # across many questions from the same video.
        if not video_ok and os.path.exists(failed_marker):
            failed += 1
            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_url": sample.video_url,
                    "error": "download_failed_cached",
                },
            )
            return

        if not video_ok:
            try:
                _download_youtube(sample.video_url, video_path, py_bin=sys.executable, timeout_s=args.yt_dlp_timeout_s)
            except Exception as e:
                failed += 1
                try:
                    with open(failed_marker, "w", encoding="utf-8") as f:
                        f.write(f"download_failed: {type(e).__name__}: {str(e)}\n")
                except Exception:
                    pass
                maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "dataset": sample.dataset,
                        "split": split,
                        "sample_id": sample.sample_id,
                        "uid": sample.uid,
                        "video_key": sample.video_key,
                        "video_url": sample.video_url,
                        "error": f"download_failed: {type(e).__name__}: {str(e)[:400]}",
                    },
                )
                return

        try:
            total_frames, fps = extract_video_info(video_path)
            tl_len = timeline_len_1fps(total_frames, fps)
        except Exception as e:
            failed += 1
            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "error": f"video_probe_failed: {type(e).__name__}: {str(e)[:400]}",
                },
            )
            return

        if tl_len <= 0:
            failed += 1
            return

        # LVBench provides `time_reference` that localizes where the evidence is in the video.
        # We bias/scope frame sampling to this window to avoid spending rounds on irrelevant frames.
        time_range = None
        if sample.dataset == "lvbench" and sample.time_reference:
            time_range = parse_time_reference_range(sample.time_reference, tl_len)
        if time_range is None:
            range_start, range_end = 0, tl_len - 1
        else:
            range_start, range_end = time_range

        if sample.dataset == "videomme" and args.videomme_use_official_prompt:
            question_block = format_videomme_question_block(sample.question, sample.options)
        else:
            question_block = format_question_block(sample.question, sample.options)
        summary_state = (
            "P: the agent has not seen any frames yet; "
            "O: no reliable observation yet; "
            "H: my belief will be updated based on what is observed; "
            "U: key detail is still unclear; "
            "R: need evidence from frames"
        )
        seen_frames: list[int] = []
        answer_letter: Optional[str] = None
        effective_rounds = 0
        terminated_invalid = False

        init_frames = sample_uniform_indices_inclusive(range_start, range_end, args.max_frames_per_round)
        next_frames = [int(i) for i in init_frames if i >= 0]

        for round_idx in range(1, args.max_rounds + 1):
            frames_this_round = [i for i in next_frames if i not in seen_frames]
            if not frames_this_round:
                frames_this_round = sample_uniform_indices_inclusive(range_start, range_end, 1)
            frames_this_round = frames_this_round[: args.max_frames_per_round]
            for i in frames_this_round:
                if i not in seen_frames:
                    seen_frames.append(i)

            # Propose candidate frames within the active window.
            local_len = max(0, range_end - range_start + 1)
            if local_len <= 0:
                candidate_next_frames = []
            else:
                seen_local = {int(i - range_start) for i in seen_frames if range_start <= i <= range_end}
                cand_local = propose_candidate_frames(
                    frame_count=local_len,
                    seen=seen_local,
                    k=int(args.candidate_k),
                    rng=rng,
                )
                candidate_next_frames = [int(i + range_start) for i in cand_local]

            images = extract_frames_1fps(video_path, frames_this_round)
            user_text = _build_user_text(
                question_block=question_block,
                summary=summary_state,
                timeline_len=tl_len,
                round_idx=round_idx,
                current_frames=frames_this_round,
                seen_frames=seen_frames,
                candidate_unseen_frames=candidate_next_frames,
                use_candidate_frame_ids=bool(args.use_candidate_frame_ids),
                require_candidate_frames=bool(args.require_candidate_frames),
                time_reference=sample.time_reference,
                num_options=len(sample.options),
            )
            if args.force_final_answer and round_idx >= args.max_rounds:
                user_text = (
                    f"{user_text}\n\n"
                    "This is the final round. You MUST answer now using <think>...</think> then "
                    "<think>...</think> then <answer>LETTER</answer>."
                )

            retry_feedback: Optional[str] = None
            raw_output = ""
            for retry_idx in range(args.max_retries_per_round + 1):
                try:
                    raw_output = _chat_once(
                        base_url=base_url,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_text=user_text if retry_feedback is None else _retry_feedback_text(
                            retry_feedback, force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds)
                        ),
                        images=images,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout_s,
                    )
                    total_model_calls += 1
                except Exception as e:
                    err_txt = f"{type(e).__name__}: {str(e)[:400]}"
                    if args.restart_server_on_failure and args.start_server:
                        if server_proc is not None:
                            stop_server(server_proc)
                        server_proc = _start_vllm_server(args)
                        wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
                        model_id = get_model_id(base_url, model_id=args.model_id)
                        retry_feedback = f"Server error; please retry. ({err_txt})"
                        total_retries += 1
                        if retry_idx < args.max_retries_per_round:
                            continue
                    failed += 1
                    maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "dataset": sample.dataset,
                            "split": split,
                            "sample_id": sample.sample_id,
                            "uid": sample.uid,
                            "video_key": sample.video_key,
                            "video_url": sample.video_url,
                            "video_path": video_path,
                            "round_idx": round_idx,
                            "error": f"server_error: {err_txt}",
                        },
                    )
                    return

                think = extract_tag(raw_output, THINK_RE)
                if think is not None:
                    think_present += 1

                summary_out = extract_tag(raw_output, SUMMARIZE_RE)
                frames_text = extract_tag(raw_output, SELECT_RE)
                answer_tag = extract_tag(raw_output, ANSWER_RE)

                if summary_out is None:
                    # Some models may omit <summarize>. Keep previous summary state and proceed if we can still parse
                    # an answer or a frame request; log this for diagnostics.
                    missing_summary += 1
                    summary_out = summary_state

                # Prefer <answer> tag; otherwise attempt to parse from text AFTER </summarize> to avoid
                # accidentally grabbing "P/O/H/U/R" from the summary.
                answer_candidate_text = answer_tag or _bare_answer_after_summary(raw_output)

                if answer_candidate_text is not None:
                    letter = normalize_answer_letter(answer_candidate_text, len(sample.options))
                    if letter is not None:
                        answer_letter = letter
                        summary_state = summary_out
                        break
                    if answer_tag is not None:
                        retry_feedback = "Invalid response: <answer> must be a single option letter."
                        invalid_outputs += 1
                        total_retries += 1
                        continue

                requested = dedupe_preserve_order(parse_int_list(frames_text))
                if not requested:
                    if frames_text is None:
                        # Recover if the model output candidate IDs but forgot the <select> tag.
                        tail = raw_output or ""
                        try:
                            m_end = None
                            for m in re.finditer(r"</summarize>", raw_output or "", flags=re.IGNORECASE):
                                m_end = m.end()
                            if m_end is not None:
                                tail = (raw_output or "")[m_end:]
                        except Exception:
                            tail = raw_output or ""
                        requested = dedupe_preserve_order(parse_int_list(tail))

                if not requested:
                    retry_feedback = (
                        "Invalid response: provide either <answer>LETTER</answer> OR <select>id1,id2</select> "
                        "after the <summarize>."
                    )
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if len(requested) > args.max_frames_per_round:
                    requested = requested[: args.max_frames_per_round]
                    invalid_outputs += 1

                mapped = requested
                if args.use_candidate_frame_ids and candidate_next_frames:
                    mapped2: list[int] = []
                    allowed = set(int(x) for x in candidate_next_frames)
                    for cid in requested:
                        if 1 <= cid <= len(candidate_next_frames):
                            mapped2.append(int(candidate_next_frames[cid - 1]))
                        elif cid in allowed:
                            # Allow direct timeline indices if they correspond to candidate frames.
                            mapped2.append(int(cid))
                    mapped = mapped2
                if args.require_candidate_frames and candidate_next_frames:
                    allowed = set(int(x) for x in candidate_next_frames)
                    if any(x not in allowed for x in mapped):
                        retry_feedback = "Invalid response: requested frames must be within candidate IDs."
                        invalid_outputs += 1
                        total_retries += 1
                        continue

                # Must be unseen.
                if any(x in seen_frames for x in mapped):
                    retry_feedback = "Invalid response: requested frames must be NEW (unseen)."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if mapped:
                    next_frames = mapped
                    effective_rounds += 1
                else:
                    next_frames = candidate_next_frames[: args.max_frames_per_round]
                    invalid_outputs += 1

                summary_state = summary_out
                break

            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_url": sample.video_url,
                    "video_path": video_path,
                    "timeline_len": tl_len,
                    "round_idx": round_idx,
                    "retry_feedback": retry_feedback,
                    "question": sample.question,
                    "options": sample.options,
                    "answer_gt": sample.answer_letter,
                    "seen_frames": seen_frames,
                    "current_frames": frames_this_round,
                    "candidate_unseen_frames": candidate_next_frames,
                    "summary_in": summary_state,
                    "raw_output": raw_output,
                    "answer_letter": answer_letter,
                },
            )

            if answer_letter is not None:
                frames_used = len(seen_frames)
                total_frames_used_all += frames_used
                total_frames_used_answered += frames_used
                answered += 1
                total_rounds += round_idx
                total_effective_rounds += effective_rounds
                gt = normalize_answer_letter(sample.answer_letter, len(sample.options))
                if gt is not None and answer_letter == gt:
                    correct += 1
                return

        # If we still have no answer, force one extra call (answer-only).
        if answer_letter is None and args.force_final_answer:
            images = extract_frames_1fps(video_path, frames_this_round)
            user_text2 = (
                f"{question_block}\n"
                "You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."
            )
            try:
                raw = _chat_once(
                    base_url=base_url,
                    model_id=model_id,
                    system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                    user_text=user_text2,
                    images=images,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                )
            except Exception as e:
                failed += 1
                maybe_log_jsonl(
                    args.log_jsonl,
                    {
                        "ts": time.time(),
                        "dataset": sample.dataset,
                        "split": split,
                        "sample_id": sample.sample_id,
                        "uid": sample.uid,
                        "video_key": sample.video_key,
                        "video_url": sample.video_url,
                        "video_path": video_path,
                        "round_idx": args.max_rounds,
                        "error": f"server_error_final_answer: {type(e).__name__}: {str(e)[:400]}",
                    },
                )
                return
            total_model_calls += 1
            ans = extract_tag(raw, ANSWER_RE)
            summary = extract_tag(raw, SUMMARIZE_RE)
            letter = normalize_answer_letter(ans or "", len(sample.options))
            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_url": sample.video_url,
                    "video_path": video_path,
                    "round_idx": int(args.max_rounds) + 1,
                    "done": letter is not None,
                    "forced_answer": True,
                    "system_prompt": FINAL_ANSWER_SYSTEM_PROMPT,
                    "user_text": user_text2,
                    "raw_output": raw,
                    "summary": summary,
                    "final_answer": letter,
                },
            )
            if letter is not None:
                answer_letter = letter
                frames_used = len(seen_frames)
                total_frames_used_all += frames_used
                total_frames_used_answered += frames_used
                answered += 1
                total_rounds += args.max_rounds
                total_effective_rounds += effective_rounds
                gt = normalize_answer_letter(sample.answer_letter, len(sample.options))
                if gt is not None and answer_letter == gt:
                    correct += 1
            else:
                terminated_invalid = True

        if terminated_invalid:
            invalid_action_terminated += 1
            total_frames_used_all += len(seen_frames)

    processed = 0
    for sample in samples:
        _process_one(sample)
        processed += 1
        if processed % 20 == 0:
            acc = correct / max(1, processed)
            avg_rounds = total_rounds / max(1, processed)
            avg_frames = total_frames_used_answered / max(1, answered)
            print(
                f"[{processed}/{len(samples)}] acc={acc:.4f} avg_rounds={avg_rounds:.3f} avg_frames={avg_frames:.2f} "
                f"failed={failed} invalid_term={invalid_action_terminated} calls={total_model_calls} "
                f"elapsed_s={time.time()-start_t:.1f}",
                flush=True,
            )
            wandb_log(
                run,
                {
                    "eval/acc": acc,
                    "eval/avg_rounds": avg_rounds,
                    "eval/avg_frames_used": avg_frames,
                    "eval/failed": failed,
                    "eval/invalid_action_terminated": invalid_action_terminated,
                    "eval/invalid_outputs": invalid_outputs,
                    "eval/total_calls": total_model_calls,
                    "eval/think_present": think_present,
                    "eval/missing_summary": missing_summary,
                },
                step=processed,
            )

    elapsed = time.time() - start_t
    acc = correct / max(1, processed)
    avg_rounds = total_rounds / max(1, processed)
    avg_effective_rounds = total_effective_rounds / max(1, processed)
    avg_frames_used = total_frames_used_answered / max(1, answered)
    avg_frames_used_all = total_frames_used_all / max(1, processed)
    prompt_log_lines = 0
    prompt_log_bytes = 0
    if args.log_jsonl and os.path.exists(args.log_jsonl):
        prompt_log_bytes = os.path.getsize(args.log_jsonl)
        with open(args.log_jsonl, "r", encoding="utf-8") as f:
            prompt_log_lines = sum(1 for _ in f)

    results = {
        "samples": processed,
        "answered": answered,
        "correct": correct,
        "accuracy": acc,
        "avg_rounds": avg_rounds,
        "avg_effective_rounds": avg_effective_rounds,
        "avg_frames_used": avg_frames_used,
        "avg_frames_used_all": avg_frames_used_all,
        "failed": failed,
        "elapsed_s": elapsed,
        "prompt_log_lines": prompt_log_lines,
        "prompt_log_bytes": prompt_log_bytes,
        "invalid_outputs": invalid_outputs,
        "invalid_action_terminated": invalid_action_terminated,
        "total_retries": total_retries,
        "total_model_calls": total_model_calls,
        "think_present_rounds": think_present,
        "missing_summary_rounds": missing_summary,
    }
    print(json.dumps(results, indent=2), flush=True)

    wandb_info: Optional[dict[str, Any]] = None
    if run is not None:
        run.summary["answered"] = answered
        run.summary["final_acc"] = acc
        run.summary["final_avg_rounds"] = avg_rounds
        run.summary["final_avg_effective_rounds"] = avg_effective_rounds
        run.summary["final_avg_frames_used"] = avg_frames_used
        run.summary["final_avg_frames_used_all"] = avg_frames_used_all
        run.summary["failed"] = failed
        run.summary["invalid_outputs"] = invalid_outputs
        run.summary["invalid_action_terminated"] = invalid_action_terminated
        run.summary["prompt_log_jsonl"] = args.log_jsonl
        run.summary["prompt_log_lines"] = prompt_log_lines
        run.summary["prompt_log_bytes"] = prompt_log_bytes
        run.summary["think_present_rounds"] = think_present
        run.summary["missing_summary_rounds"] = missing_summary
        run.finish()
        wandb_info = {
            "enabled": True,
            "mode": getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE"),
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "group": args.wandb_group,
            "id": getattr(run, "id", None),
            "url": getattr(run, "url", None),
            "run_dir": getattr(run, "dir", None),
        }

    if args.summary_json:
        out_dir = os.path.dirname(args.summary_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **run_config,
                    "results": results,
                    "prompt_log_jsonl": args.log_jsonl,
                    "wandb": wandb_info,
                    "command": " ".join(["python", *sys.argv]),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    if server_proc is not None and args.start_server:
        stop_server(server_proc)
    if processed > 0 and (failed >= processed or total_model_calls == 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
