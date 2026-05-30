#!/usr/bin/env python3
"""REVISE multi-round plug-and-play evaluation for EgoSchema / VideoEspresso (vLLM).

Runs the REVISE question-aware sparse-video loop against a vLLM
OpenAI-compatible server for EgoSchema (and the VideoEspresso multiple-choice
split, which shares this loader and loop). Each round the model reasons in
``<think>``, emits a ``<summarize>`` P/O/H/U/R state, then either ``<select>``s
previously unseen frames or commits an ``<answer>``; a final answer is
force-requested once ``--max-rounds`` is reached.

Includes component-ablation switches (e.g. omitting the carried summary state,
or accepting an unstructured summary) used by the paper's ablations. Run as a
CLI (see ``main``); invoked by ``run_generate_teacher_data_videoespresso.sh``
and ``scripts/paper_suite.py``. Shared helpers live in ``revise/pnp_utils.py``.
"""

from __future__ import annotations

import argparse
import hashlib
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

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


_EGOSCHEMA_CHUNK_CACHE: dict[tuple[str, str], str | None] = {}

FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are a multiple-choice video QA assistant. "
    "Return exactly <think>...</think> followed by <answer>LETTER</answer>. "
    "LETTER must be one of the option letters in the question. "
    "Do not output <summarize>, <select>, requests, or any text outside these tags."
)

UNSTRUCTURED_SYSTEM_PROMPT = (
    "You are REVISE, a multi-round video reasoning agent.\n"
    "Each round you will see a multiple-choice question, a few sampled video frames, and possibly a compact summary.\n"
    "If you are confident, answer the question. If you are not confident, request more video frames to view next.\n"
    "Frames are sampled at ~1 fps; frame index is approximately the timestamp in seconds.\n\n"
    "IMPORTANT: Output must follow exactly one of the two formats below. Do not output text outside tags.\n"
    "EVERY response MUST begin with a <think>...</think> reasoning trace.\n"
    "On a Select round, follow <think> with <summarize> then <select>.\n"
    "On the Answer round, follow <think> with <answer> only (reuse your last committed summary).\n"
    "Do NOT output bare placeholders like '...', 'none', or 'N/A' as the summary.\n\n"
    "Format 1 - Request more frames:\n"
    "<think>...</think>\n"
    "<summarize>A concise natural-language summary of what is known, what remains uncertain, and what evidence is needed.</summarize>\n"
    "<select>1, 3</select>\n\n"
    "Format 2 - Answer now:\n"
    "<think>...</think>\n"
    "<answer>B</answer>\n\n"
    "Rules:\n"
    "- Every response begins with <think>...</think>.\n"
    "- Frame indices are 0-based in [0, L-1].\n"
    "- If requesting, choose 1 to {max_frames_per_round} NEW frames to view NEXT.\n"
    "- Do NOT output any frame index from the Seen frames list; those are already viewed.\n"
    "- In <select>, output comma-separated integers only.\n"
    "- In <answer>, output EXACTLY ONE option letter shown in the question.\n"
)

from revise.pnp_prompts import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from revise.pnp_utils import (
    ANSWER_RE,
    SELECT_RE,
    SUMMARIZE_RE,
    THINK_RE,
    b64_jpeg,
    contains_banned_example,
    dedupe_preserve_order,
    ensure_writable_hf_cache,
    extract_frames,
    extract_tag,
    format_intervals,
    format_frame_list,
    format_videoespresso_question_block,
    get_api_headers,
    get_model_id,
    in_intervals,
    indices_to_intervals,
    is_placeholder,
    maybe_init_wandb,
    normalize_answer_letter,
    parse_int_list,
    propose_candidate_frames,
    resolve_base_url,
    sample_uniform_indices,
    build_vllm_serve_command,
    open_server_log_streams,
    stop_server,
    summary_has_ohrpu,
    unseen_intervals,
    wait_port,
    wait_for_server,
)


def _parse_answer_idx(raw_answer: Any, num_choices: int) -> Optional[int]:
    if raw_answer is None:
        return None
    text = str(raw_answer).strip()
    if not text or text.lower() == "none":
        return None
    m = re.search(r"([A-E])", text.upper())
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx < num_choices:
            return idx
        return None
    try:
        idx = int(float(text))
    except Exception:
        return None
    if 0 <= idx < num_choices:
        return idx
    if 1 <= idx <= num_choices:
        return idx - 1
    return None


def _normalize_option_text(opt: str) -> str:
    # EgoSchema options may already start with "(A):". Strip that to avoid double labels.
    t = str(opt).strip()
    t = re.sub(r"^\(?[A-E]\)?\s*[:.)-]\s*", "", t)
    return t.strip()


def _format_question(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if labels:
        lines.append(
            "To answer, output <think>...</think> then <answer>LETTER</answer> "
            f"(LETTER must be one of: {', '.join(labels)})."
        )
    return "\n".join(lines)


def _system_prompt_for_mode(*, structured_summary: bool, max_frames_per_round: int) -> str:
    template = DEFAULT_SYSTEM_PROMPT if structured_summary else UNSTRUCTURED_SYSTEM_PROMPT
    return template.format(max_frames_per_round=max_frames_per_round)


def _summary_is_valid_for_mode(summary_text: Optional[str], *, require_structured: bool) -> bool:
    if summary_text is None or is_placeholder(summary_text) or contains_banned_example(summary_text):
        return False
    if require_structured and not summary_has_ohrpu(summary_text):
        return False
    return True


def _build_user_text(
    question_block: str,
    summary: str,
    frame_count: int,
    round_idx: int,
    frame_indices: list[int],
    seen_frames: list[int],
    *,
    hide_seen_frames: bool = False,
    candidate_unseen_frames: Optional[list[int]] = None,
    use_candidate_frame_ids: bool = False,
    require_candidate_frames: bool = False,
    carry_summary_state: bool = True,
) -> str:
    lines: list[str] = [f"Round {round_idx} / Question:\n{question_block}", f"Total frames L = {frame_count}."]
    if hide_seen_frames:
        lines.append(
            f"Seen frames: {len(seen_frames)} frames already viewed "
            "(do NOT request any previously shown frames; follow the selection constraints below)."
        )
    else:
        lines.append(f"Seen frames (already viewed; do NOT request these again): {format_frame_list(seen_frames)}")

    if candidate_unseen_frames and use_candidate_frame_ids:
        lines.append(
            "Candidate unseen frames available as IDs (all NEW): "
            f"choose IDs in [1, {len(candidate_unseen_frames)}]."
        )
        lines.append(
            "In <select>, output ONLY candidate IDs (comma-separated). Do NOT output raw frame indices when IDs exist."
        )
    else:
        lines.append(
            "Allowed unseen frame ranges for <select> (choose NEW indices only from these ranges): "
            f"{format_intervals(unseen_intervals(frame_count, seen_frames))}"
        )
        if candidate_unseen_frames:
            prefix = (
                "Candidate unseen frame ranges to request (REQUIRED, all NEW): "
                if require_candidate_frames
                else "Candidate unseen frame ranges to request (optional, all NEW): "
            )
            lines.append(prefix + f"{format_intervals(indices_to_intervals(candidate_unseen_frames))}")
            if require_candidate_frames:
                lines.append("In <select>, output ONLY indices within the Candidate unseen frame ranges above.")
    if carry_summary_state:
        lines.extend(["Current summary:", f"<summarize>{summary}</summarize>"])
    else:
        lines.append(
            "Current summary: state carryover is disabled for this ablation; "
            "use only the question, seen-frame count/ranges, and frames shown in this round."
        )
    lines.append("Frames shown in this round:")
    if hide_seen_frames or (candidate_unseen_frames and use_candidate_frame_ids):
        for i, _ in enumerate(frame_indices):
            label = chr(ord("A") + i)
            lines.append(f"Shown frame {label} <image>")
    else:
        for idx in frame_indices:
            lines.append(f"Frame {idx} <image>")
    return "\n".join(lines)


def _sample_unseen_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []
    candidates = [i for i in range(frame_count) if i not in seen]
    rng.shuffle(candidates)
    return sorted(candidates[:k])


def _call_chat_completions(
    base_url: str,
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=get_api_headers(),
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data["choices"][0]["message"]["content"])


def _start_vllm_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    cmd, env = build_vllm_serve_command(
        args,
        image_limit=int(args.max_frames_per_round),
        cuda_visible_default="0,1,2,3",
    )
    server_stdout, server_stderr = open_server_log_streams(args.server_log)
    return subprocess.Popen(cmd, env=env, stdout=server_stdout, stderr=server_stderr)


def _stable_sample_id(video_path: str, question: str, choices: list[str], answer_idx: int) -> str:
    payload = {
        "video_path": str(video_path),
        "question": str(question),
        "choices": [str(c) for c in (choices or [])],
        "answer_idx": int(answer_idx),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass
class EgoSchemaSample:
    sample_id: str
    qid: str
    video_path: str
    question: str
    choices: list[str]
    answer_idx: int
    frame_count: int
    task: str = ""
    evidence: str = ""


def _resolve_local_video_path(video_root: str, rel_path: str) -> str | None:
    candidates = [
        os.path.join(video_root, rel_path),
        os.path.join(video_root, "all_video", rel_path),
        os.path.join(video_root, "train_video", "all_video", rel_path),
        os.path.join(video_root, "test_video", "all_video", rel_path),
        os.path.join(video_root, "videos", "videos", os.path.basename(rel_path)),
    ]
    for video_path in candidates:
        if os.path.exists(video_path):
            return video_path
    return None


def _download_egoschema_video(
    *,
    video_root: str,
    video_filename: str,
    repo_id: str,
) -> str | None:
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files
        import fsspec
        import zipfile
    except Exception:
        return None
    os.makedirs(video_root, exist_ok=True)
    target_path = Path(video_root) / video_filename
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=video_filename,
            local_dir=video_root,
        )
        if path and os.path.exists(path):
            return str(path)
    except Exception:
        pass

    chunk_repo = "VLM2Vec/egoschema"
    cache_key = (chunk_repo, video_filename)
    chunk_name = _EGOSCHEMA_CHUNK_CACHE.get(cache_key)
    if chunk_name is None and cache_key not in _EGOSCHEMA_CHUNK_CACHE:
        try:
            chunk_names = sorted(
                f
                for f in list_repo_files(chunk_repo, repo_type="dataset")
                if re.fullmatch(r"videos_chunked_\d+\.zip", os.path.basename(f))
            )
            member_name = f"videos/{video_filename}"
            for candidate in chunk_names:
                url = hf_hub_url(chunk_repo, filename=candidate, repo_type="dataset")
                with fsspec.open(url, "rb", block_size=2 * 1024 * 1024) as remote_f:
                    with zipfile.ZipFile(remote_f) as zf:
                        if member_name in set(zf.namelist()):
                            chunk_name = candidate
                            break
            _EGOSCHEMA_CHUNK_CACHE[cache_key] = chunk_name
        except Exception:
            _EGOSCHEMA_CHUNK_CACHE[cache_key] = None
            chunk_name = None
    if not chunk_name:
        return None
    try:
        member_name = f"videos/{video_filename}"
        url = hf_hub_url(chunk_repo, filename=chunk_name, repo_type="dataset")
        with fsspec.open(url, "rb", block_size=2 * 1024 * 1024) as remote_f:
            with zipfile.ZipFile(remote_f) as zf:
                with zf.open(member_name) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
    except Exception:
        return None
    return str(target_path) if target_path.exists() else None


def _rows_to_egoschema_samples(
    rows: list[dict[str, Any]],
    video_root: str,
    *,
    allow_video_download: bool = False,
    egoschema_video_repo: str = "VLM2Vec/egoschema-rawvideo",
) -> list[EgoSchemaSample]:
    samples: list[EgoSchemaSample] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_idx") or row.get("qid") or row.get("id") or "")
        question = str(row.get("question") or "")
        options = row.get("options") or row.get("choices") or []
        if not isinstance(options, list) or len(options) < 2:
            continue
        choices = [_normalize_option_text(o) for o in options]
        answer_idx = _parse_answer_idx(row.get("correct_answer") or row.get("answer"), len(choices))
        if answer_idx is None:
            continue
        rel = str(row.get("video_path") or row.get("video") or "")
        if not rel:
            continue
        video_path = _resolve_local_video_path(video_root, rel)
        if video_path is None and allow_video_download:
            video_path = _download_egoschema_video(
                video_root=video_root,
                video_filename=os.path.basename(rel),
                repo_id=egoschema_video_repo,
            )
        if video_path is None:
            continue

        sample_id = _stable_sample_id(video_path, question, choices, answer_idx)
        samples.append(
            EgoSchemaSample(
                sample_id=sample_id,
                qid=qid,
                video_path=video_path,
                question=question,
                choices=choices,
                answer_idx=answer_idx,
                frame_count=0,
                task=str(row.get("task") or ""),
                evidence=str(row.get("evidence") or ""),
            )
        )
    return samples


def _load_egoschema_samples(
    json_path: str,
    video_root: str,
    max_samples: int,
    seed: int,
    *,
    allow_video_download: bool = False,
    egoschema_video_repo: str = "VLM2Vec/egoschema-rawvideo",
) -> list[EgoSchemaSample]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {json_path}, got {type(data)}")

    rng = random.Random(seed)
    if max_samples > 0 and len(data) > max_samples:
        rng.shuffle(data)
        data = data[:max_samples]
    return _rows_to_egoschema_samples(
        data,
        video_root,
        allow_video_download=allow_video_download,
        egoschema_video_repo=egoschema_video_repo,
    )


def _load_egoschema_hf_samples(
    *,
    video_root: str,
    max_samples: int,
    seed: int,
    hf_config: str,
    allow_video_download: bool,
    egoschema_video_repo: str,
) -> list[EgoSchemaSample]:
    ensure_writable_hf_cache(REPO_ROOT / "data" / "revise_assets" / "hf_home")
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for EgoSchema HF fallback.") from exc

    ds = load_dataset("VLM2Vec/egoschema", hf_config, split="test")
    rows = list(ds)
    rng = random.Random(seed)
    if max_samples > 0 and len(rows) > max_samples:
        rng.shuffle(rows)
        rows = rows[:max_samples]

    canon_rows: list[dict[str, Any]] = []
    for row in rows:
        options = row.get("option") or row.get("options") or []
        if not isinstance(options, list):
            continue
        answer_idx = _parse_answer_idx(row.get("answer"), len(options))
        if answer_idx is None:
            continue
        canon_rows.append(
            {
                "question_idx": row.get("question_idx"),
                "question": row.get("question"),
                "options": options,
                "correct_answer": chr(ord("A") + int(answer_idx)),
                "video_path": f"{row.get('video_idx')}.mp4",
            }
        )

    return _rows_to_egoschema_samples(
        canon_rows,
        video_root,
        allow_video_download=allow_video_download,
        egoschema_video_repo=egoschema_video_repo,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF model id or local snapshot path")
    parser.add_argument("--video-root", default=None, help="Root directory containing videos referenced by JSON")
    parser.add_argument(
        "--json",
        default=None,
        help="Local MC video-QA JSON list (e.g., EgoSchema subset or VideoEspresso bench_hard.json).",
    )
    parser.add_argument(
        "--dataset-name",
        default="egoschema",
        help="Dataset label for summaries/logging (e.g., egoschema, videoespresso).",
    )
    parser.add_argument(
        "--egoschema-source",
        choices=["auto", "local", "hf"],
        default="auto",
        help="For EgoSchema only: use local assets, Hugging Face fallback, or auto-detect.",
    )
    parser.add_argument(
        "--egoschema-hf-config",
        default="Subset",
        help="HF split config for EgoSchema fallback (default: Subset).",
    )
    parser.add_argument(
        "--egoschema-video-cache-dir",
        default=str(REPO_ROOT / "outputs" / "egoschema_hf" / "videos"),
        help="Cache directory for EgoSchema videos downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--auto-download-egoschema-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using EgoSchema HF fallback, download missing mp4s on demand.",
    )
    parser.add_argument(
        "--egoschema-video-repo",
        default="VLM2Vec/egoschema-rawvideo",
        help="HF dataset repo containing EgoSchema mp4 files.",
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument(
        "--videoespresso-use-official-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For dataset-name=videoespresso, format task/options prompts like the official close-ended evaluator.",
    )
    parser.add_argument(
        "--videoespresso-with-evidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For dataset-name=videoespresso, include the row evidence field in the prompt.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Shard the dataset for data-parallel evaluation.")
    parser.add_argument("--shard-idx", type=int, default=0, help="Shard index in [0, num_shards).")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-frames-per-round", "--max-frames", type=int, default=5)
    parser.add_argument(
        "--use-candidate-frames",
        action="store_true",
        help="Include a small list of candidate unseen frame indices in the prompt to help frame selection.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=0,
        help="Number of candidate unseen frames to propose when --use-candidate-frames is set (default: max(12, max_frames_per_round*4)).",
    )
    parser.add_argument(
        "--use-candidate-frame-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --use-candidate-frames is enabled, expose candidates as IDs 1..K (not raw indices) and require <select> to output candidate IDs.",
    )
    parser.add_argument(
        "--require-candidate-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When candidate frames are provided, treat them as an allowlist: <select> must only contain indices within the candidate unseen ranges.",
    )
    parser.add_argument(
        "--hide-seen-frames-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not print explicit seen frame indices in the prompt (reduces copying); rely on unseen ranges instead.",
    )
    parser.add_argument(
        "--ablate-state-carryover",
        action="store_true",
        help="Component ablation: do not carry the previous <summarize> state into the next-round prompt.",
    )
    parser.add_argument(
        "--ablate-structured-summary",
        action="store_true",
        help="Component ablation: allow unstructured natural-language <summarize> text instead of requiring P/O/H/U/R fields.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Defaults to http://host:port.")
    parser.add_argument("--model-id", default=None, help="Explicit remote model ID for chat completions.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--start-server", action="store_true", help="Start vLLM server subprocess")
    parser.add_argument(
        "--restart-server-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --start-server, restart vLLM if a request fails (e.g., timeout).",
    )
    parser.add_argument("--server-log", type=str, default=None, help="Optional file path to append vLLM server logs")
    parser.add_argument("--server-timeout-s", type=int, default=300)
    parser.add_argument("--request-timeout-s", type=int, default=300)
    parser.add_argument("--max-retries-per-round", type=int, default=2)
    parser.add_argument(
        "--strict-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate the sample immediately on illegal actions (e.g., invalid <select>, missing required tags, <think>).",
    )
    parser.add_argument("--log-jsonl", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None, help="Optional path to save a run summary JSON.")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument(
        "--resume-from-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If log-jsonl exists, skip already-completed samples and continue appending to the same log.",
    )
    parser.add_argument(
        "--force-final-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no <answer> is produced within max rounds, issue a final answer-only request.",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Log eval metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)

    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError("--shard-idx must be in [0, num_shards)")
    if args.base_url and args.start_server:
        raise ValueError("--base-url cannot be combined with --start-server.")
    if not args.wandb_project:
        args.wandb_project = f"revise_{str(args.dataset_name).strip().lower() or 'local_mc'}"

    rng = random.Random(args.seed)
    dataset_name = str(args.dataset_name).strip().lower()

    use_hf_egoschema = (
        dataset_name == "egoschema"
        and (
            args.egoschema_source == "hf"
            or (args.egoschema_source == "auto" and (not args.json or not args.video_root))
        )
    )
    if use_hf_egoschema:
        args.video_root = args.video_root or args.egoschema_video_cache_dir
        samples = _load_egoschema_hf_samples(
            video_root=str(args.video_root),
            max_samples=args.max_samples,
            seed=args.seed,
            hf_config=str(args.egoschema_hf_config),
            allow_video_download=bool(args.auto_download_egoschema_videos),
            egoschema_video_repo=str(args.egoschema_video_repo),
        )
    else:
        if not args.json or not args.video_root:
            raise ValueError("--json and --video-root are required for local JSON datasets.")
        samples = _load_egoschema_samples(
            args.json,
            args.video_root,
            args.max_samples,
            args.seed,
            allow_video_download=False,
            egoschema_video_repo=str(args.egoschema_video_repo),
        )
    if args.num_shards > 1:
        samples = [s for i, s in enumerate(samples) if (i % args.num_shards) == args.shard_idx]
    if not samples:
        raise RuntimeError("No samples loaded (check dataset source, JSON, video cache, or HF video availability).")

    if args.candidate_k <= 0:
        args.candidate_k = max(12, args.max_frames_per_round * 4)

    os.makedirs("debug_runs", exist_ok=True)
    log_jsonl = args.log_jsonl
    if log_jsonl:
        os.makedirs(os.path.dirname(log_jsonl) or ".", exist_ok=True)

    summary: dict[str, Any] = {
        "task": f"revise_plug_and_play_{str(args.dataset_name).strip().lower()}_vllm",
        "dataset_name": args.dataset_name,
        "dataset_json": args.json,
        "video_root": args.video_root,
        "egoschema_source": args.egoschema_source if dataset_name == "egoschema" else "local",
        "egoschema_hf_config": args.egoschema_hf_config if dataset_name == "egoschema" else None,
        "auto_download_egoschema_videos": bool(args.auto_download_egoschema_videos) if dataset_name == "egoschema" else False,
        "egoschema_video_repo": args.egoschema_video_repo if dataset_name == "egoschema" else None,
        "model_path": args.model_path,
        "engine": "vllm",
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_samples": args.max_samples,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "max_rounds": args.max_rounds,
        "max_frames_per_round": args.max_frames_per_round,
        "max_retries_per_round": args.max_retries_per_round,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "use_candidate_frames": bool(args.use_candidate_frames),
        "candidate_k": int(args.candidate_k),
        "use_candidate_frame_ids": bool(args.use_candidate_frame_ids),
        "require_candidate_frames": bool(args.require_candidate_frames),
        "hide_seen_frames_in_prompt": bool(args.hide_seen_frames_in_prompt),
        "ablate_state_carryover": bool(args.ablate_state_carryover),
        "ablate_structured_summary": bool(args.ablate_structured_summary),
        "videoespresso_use_official_prompt": bool(args.videoespresso_use_official_prompt)
        if dataset_name == "videoespresso"
        else None,
        "videoespresso_with_evidence": bool(args.videoespresso_with_evidence)
        if dataset_name == "videoespresso"
        else None,
    }

    wandb_run = maybe_init_wandb(args, summary)

    server_proc: Optional[subprocess.Popen[str]] = None
    base_url = resolve_base_url(args.base_url, args.host, args.port)
    model_id: Optional[str] = None
    structured_summary = not bool(args.ablate_structured_summary)
    system_prompt = _system_prompt_for_mode(
        structured_summary=structured_summary,
        max_frames_per_round=args.max_frames_per_round,
    )

    def _ensure_server() -> None:
        nonlocal server_proc, model_id
        if args.start_server and server_proc is None:
            server_proc = _start_vllm_server(args)
            wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
            wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
        if model_id is None:
            model_id = get_model_id(base_url, model_id=args.model_id)

    def _restart_server() -> None:
        nonlocal server_proc, model_id
        if server_proc is not None:
            stop_server(server_proc)
            server_proc = None
            model_id = None
        if args.start_server:
            server_proc = _start_vllm_server(args)
            wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
            wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
            model_id = get_model_id(base_url, model_id=args.model_id)

    # Resume logic
    done_ids: set[str] = set()
    if log_jsonl and args.resume_from_log and os.path.exists(log_jsonl):
        with open(log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sid = str(rec.get("sample_id") or "")
                if sid and rec.get("done"):
                    done_ids.add(sid)

    # Eval loop
    correct = 0
    total = 0
    failed = 0
    total_rounds = 0
    invalid_outputs = 0
    total_retries = 0
    total_model_calls = 0
    fallback_frames_used = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        if sample.sample_id in done_ids:
            continue

        total += 1
        if dataset_name == "videoespresso" and args.videoespresso_use_official_prompt:
            question_block = format_videoespresso_question_block(
                sample.question,
                sample.choices,
                task=sample.task,
                evidence=sample.evidence,
                with_evidence=bool(args.videoespresso_with_evidence),
                revise_answer_tags=True,
            )
        else:
            question_block = _format_question(sample.question, sample.choices)
        summary_state = "P: none yet; O: no observations yet; H: no belief yet; U: key details are unknown; R: need frames"
        seen_frames: list[int] = []
        used_images: list[Image.Image] = []
        final_answer: Optional[str] = None
        answer_round: Optional[int] = None
        illegal_action = False
        per_sample_invalid = 0
        per_sample_retries = 0

        # Compute frame_count lazily.
        frame_count = sample.frame_count
        if frame_count <= 0:
            try:
                import decord

                vr = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))
                frame_count = int(len(vr))
            except Exception:
                frame_count = 0

        init_frames = sample_uniform_indices(frame_count, args.max_frames_per_round)

        for round_idx in range(1, args.max_rounds + 1):
            if round_idx == 1:
                frames_this_round = init_frames
            else:
                if not frames_this_round:
                    frames_this_round = sample_uniform_indices(frame_count, 1)

            images = extract_frames(sample.video_path, frames_this_round)
            if not images:
                # fallback: if decoding fails, skip sample
                failed += 1
                illegal_action = True
                break

            used_images.extend(images)
            seen_frames = dedupe_preserve_order(seen_frames + frames_this_round)

            candidate_unseen = None
            if args.use_candidate_frames:
                candidate_unseen = propose_candidate_frames(
                    frame_count, set(seen_frames), args.candidate_k, rng=rng
                )

            user_text = _build_user_text(
                question_block,
                summary_state,
                frame_count,
                round_idx,
                frame_indices=frames_this_round,
                seen_frames=seen_frames,
                hide_seen_frames=args.hide_seen_frames_in_prompt,
                candidate_unseen_frames=candidate_unseen,
                use_candidate_frame_ids=args.use_candidate_frame_ids,
                require_candidate_frames=args.require_candidate_frames,
                carry_summary_state=not bool(args.ablate_state_carryover),
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg(img)}"},
                            }
                            for img in images
                        ],
                    ],
                },
            ]

            try:
                _ensure_server()
                assert model_id is not None
                raw = _call_chat_completions(
                    base_url,
                    model_id,
                    messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout_s=args.request_timeout_s,
                )
                total_model_calls += 1
            except Exception as e:
                if args.restart_server_on_failure and isinstance(e, requests.exceptions.RequestException):
                    _restart_server()
                    try:
                        _ensure_server()
                        assert model_id is not None
                        raw = _call_chat_completions(
                            base_url,
                            model_id,
                            messages,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens,
                            timeout_s=args.request_timeout_s,
                        )
                        total_model_calls += 1
                    except Exception as retry_exc:
                        if log_jsonl:
                            with open(log_jsonl, "a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "sample_id": sample.sample_id,
                                            "qid": sample.qid,
                                            "video_path": sample.video_path,
                                            "question": sample.question,
                                            "options": sample.choices,
                                            "task": sample.task,
                                            "evidence": sample.evidence,
                                            "dataset_name": dataset_name,
                                            "round": round_idx,
                                            "done": True,
                                            "error": f"{type(retry_exc).__name__}: {retry_exc}",
                                            "error_stage": "model_call_retry",
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                        failed += 1
                        illegal_action = True
                        break
                else:
                    if log_jsonl:
                        with open(log_jsonl, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "sample_id": sample.sample_id,
                                        "qid": sample.qid,
                                        "video_path": sample.video_path,
                                        "question": sample.question,
                                        "options": sample.choices,
                                        "task": sample.task,
                                        "evidence": sample.evidence,
                                        "dataset_name": dataset_name,
                                        "round": round_idx,
                                        "done": True,
                                        "error": f"{type(e).__name__}: {e}",
                                        "error_stage": "model_call",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    failed += 1
                    illegal_action = True
                    break

            answer_text = extract_tag(raw, ANSWER_RE)
            frames_text = extract_tag(raw, SELECT_RE)
            summary_text = extract_tag(raw, SUMMARIZE_RE)
            think_text = extract_tag(raw, THINK_RE)

            def _log_event(done: bool, **extra: Any) -> None:
                if not log_jsonl:
                    return
                rec = {
                    "sample_id": sample.sample_id,
                    "qid": sample.qid,
                    "video_path": sample.video_path,
                    "question": sample.question,
                    "options": sample.choices,
                    "answer_gt": chr(ord("A") + int(sample.answer_idx)),
                    "task": sample.task,
                    "evidence": sample.evidence,
                    "dataset_name": dataset_name,
                    "videoespresso_official_prompt": bool(
                        dataset_name == "videoespresso" and args.videoespresso_use_official_prompt
                    ),
                    "videoespresso_with_evidence": bool(
                        dataset_name == "videoespresso" and args.videoespresso_with_evidence
                    ),
                    "round": round_idx,
                    "done": bool(done),
                    "system_prompt": system_prompt,
                    "user_text": user_text,
                    "shown_frames": frames_this_round,
                    "seen_frames": seen_frames,
                    "candidate_unseen": candidate_unseen,
                    "ablate_state_carryover": bool(args.ablate_state_carryover),
                    "ablate_structured_summary": bool(args.ablate_structured_summary),
                    "raw_output": raw,
                    **extra,
                }
                with open(log_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Strict protocol checks + retry loop
            retries_left = int(args.max_retries_per_round)
            while True:
                invalid = False
                invalid_reason = None
                # Paper protocol: every response MUST begin with a <think> trace.
                if think_text is None:
                    invalid = True
                    invalid_reason = "missing_think"
                if answer_text:
                    # Strict-paper Answer round: <think> + <answer> only; no <summarize> required.
                    norm = normalize_answer_letter(answer_text, len(sample.choices))
                    if norm is None:
                        invalid = True
                        invalid_reason = "invalid_answer"
                    else:
                        final_answer = norm
                        answer_round = round_idx
                else:
                    # Select round: require a valid <summarize> and a <select> request.
                    if not _summary_is_valid_for_mode(summary_text, require_structured=structured_summary):
                        invalid = True
                        invalid_reason = "invalid_summary"
                    elif frames_text is None:
                        invalid = True
                        invalid_reason = "missing_frames"

                if not invalid:
                    break

                per_sample_invalid += 1
                invalid_outputs += 1
                if retries_left <= 0:
                    illegal_action = True
                    if args.strict_actions:
                        _log_event(True, illegal_action=True, invalid_reason=invalid_reason, retries_used=per_sample_retries)
                        break
                    break
                retries_left -= 1
                per_sample_retries += 1
                total_retries += 1
                feedback = (
                    "Invalid response: every response MUST begin with <think>...</think>. "
                    "On a Select round, follow <think> with <summarize> then <select>; "
                    "on the Answer round, follow <think> with <answer> only. "
                    + (
                        "Summary must contain P/O/H/U/R in order, with meaningful text (no placeholders). "
                        if structured_summary
                        else "Summary must be meaningful natural language, with no empty placeholder text. "
                    )
                    +
                    "If answering, <answer> must be exactly one letter (A/B/C/D/E)."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": feedback})
                try:
                    _ensure_server()
                    assert model_id is not None
                    raw = _call_chat_completions(
                        base_url,
                        model_id,
                        messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.request_timeout_s,
                    )
                    total_model_calls += 1
                except Exception:
                    illegal_action = True
                    break

                answer_text = extract_tag(raw, ANSWER_RE)
                frames_text = extract_tag(raw, SELECT_RE)
                summary_text = extract_tag(raw, SUMMARIZE_RE)
                think_text = extract_tag(raw, THINK_RE)

            if illegal_action:
                break

            if summary_text and not args.ablate_state_carryover:
                summary_state = summary_text.strip()

            if final_answer is not None:
                _log_event(True, final_answer=final_answer, retries_used=per_sample_retries, invalid_outputs=per_sample_invalid)
                break

            # Frame request path.
            assert frames_text is not None
            requested = dedupe_preserve_order(parse_int_list(frames_text))
            if args.use_candidate_frame_ids and candidate_unseen:
                mapped: list[int] = []
                invalid_id = False
                for cid in requested:
                    if 1 <= cid <= len(candidate_unseen):
                        mapped.append(int(candidate_unseen[cid - 1]))
                    else:
                        invalid_id = True
                if invalid_id:
                    per_sample_invalid += 1
                    invalid_outputs += 1
                    illegal_action = True if args.strict_actions else False
                    break
                requested = dedupe_preserve_order(mapped)
            elif args.require_candidate_frames and candidate_unseen:
                allowed = {int(i) for i in candidate_unseen}
                requested = [i for i in requested if int(i) in allowed]
            if not requested:
                per_sample_invalid += 1
                invalid_outputs += 1
                illegal_action = True if args.strict_actions else False
                break

            allowed_ranges = unseen_intervals(frame_count, seen_frames)
            next_frames = [i for i in requested if 0 <= i < frame_count and i not in seen_frames and in_intervals(i, allowed_ranges)]
            if not next_frames:
                fallback_frames_used += 1
                # fallback: sample a new unseen frame
                fallback = _sample_unseen_frames(frame_count, set(seen_frames), args.max_frames_per_round, rng=rng)
                next_frames = fallback[: args.max_frames_per_round] if fallback else sample_uniform_indices(frame_count, 1)
            frames_this_round = next_frames[: args.max_frames_per_round]

            _log_event(False, requested_frames=requested, next_frames=frames_this_round, retries_used=per_sample_retries)

        # End per-sample
        if final_answer is None:
            if args.force_final_answer and not illegal_action:
                # Final forced answer-only prompt (no images).
                user_text = (
                    f"Final round: answer now.\n{question_block}\n"
                    f"Total frames L = {frame_count}.\n"
                    "You must answer now.\n"
                    "Output ONLY <think>...</think> then <answer>LETTER</answer>."
                )
                messages = [
                    {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ]
                try:
                    _ensure_server()
                    assert model_id is not None
                    raw = _call_chat_completions(
                        base_url,
                        model_id,
                        messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.request_timeout_s,
                    )
                    total_model_calls += 1
                    answer_text = extract_tag(raw, ANSWER_RE)
                    final_answer = normalize_answer_letter(answer_text or "", len(sample.choices))
                    if log_jsonl:
                        rec = {
                            "sample_id": sample.sample_id,
                            "qid": sample.qid,
                            "video_path": sample.video_path,
                            "round": int(args.max_rounds) + 1,
                            "done": final_answer is not None,
                            "forced_answer": True,
                            "system_prompt": FINAL_ANSWER_SYSTEM_PROMPT,
                            "user_text": user_text,
                            "raw_output": raw,
                            "final_answer": final_answer,
                        }
                        with open(log_jsonl, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if final_answer is not None:
                        answer_round = int(args.max_rounds)
                except Exception as e:
                    if log_jsonl:
                        rec = {
                            "sample_id": sample.sample_id,
                            "qid": sample.qid,
                            "video_path": sample.video_path,
                            "round": int(args.max_rounds) + 1,
                            "done": False,
                            "forced_answer": True,
                            "system_prompt": FINAL_ANSWER_SYSTEM_PROMPT,
                            "user_text": user_text,
                            "error": repr(e),
                        }
                        with open(log_jsonl, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    final_answer = None

        if final_answer is None:
            if not illegal_action:
                failed += 1
        else:
            if final_answer == chr(ord("A") + int(sample.answer_idx)):
                correct += 1
            if answer_round is not None:
                total_rounds += int(min(int(answer_round), int(args.max_rounds)))
        if total % args.progress_interval == 0:
            acc = correct / max(1, total - failed)
            avg_r = total_rounds / max(1, total - failed)
            msg = f"[progress] {total}/{len(samples)} done | acc={acc:.3f} avg_rounds={avg_r:.2f} failed={failed} invalid={invalid_outputs} calls={total_model_calls}"
            print(msg, flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "progress/samples": total,
                        "progress/failed": failed,
                        "metrics/accuracy": acc,
                        "metrics/avg_rounds": avg_r,
                        "debug/invalid_outputs": invalid_outputs,
                        "debug/total_model_calls": total_model_calls,
                    }
                )

    elapsed_s = time.time() - start_time
    denom = max(1, total)
    acc = correct / max(1, total - failed) if total > failed else 0.0
    avg_rounds = total_rounds / max(1, total - failed) if total > failed else 0.0

    results = {
        "samples": total,
        "accuracy": acc,
        "avg_rounds": avg_rounds,
        "failed": failed,
        "elapsed_s": elapsed_s,
        "invalid_outputs": invalid_outputs,
        "total_retries": total_retries,
        "total_model_calls": total_model_calls,
        "fallback_frames_used": fallback_frames_used,
    }
    summary["results"] = results
    summary["log_jsonl"] = log_jsonl

    if log_jsonl and os.path.exists(log_jsonl):
        try:
            with open(log_jsonl, "r", encoding="utf-8") as f:
                prompt_lines = sum(1 for _ in f)
            prompt_bytes = os.path.getsize(log_jsonl)
        except Exception:
            prompt_lines = None
            prompt_bytes = None
        summary["prompt_log_lines"] = prompt_lines
        summary["prompt_log_bytes"] = prompt_bytes

    if wandb_run is not None:
        summary["wandb"] = {
            "enabled": True,
            "mode": wandb_run.settings.mode,
            "project": wandb_run.project,
            "entity": getattr(wandb_run, "entity", None),
            "name": wandb_run.name,
            "group": getattr(wandb_run, "group", None),
            "id": wandb_run.id,
            "run_dir": wandb_run.dir,
            "url": getattr(wandb_run, "url", None),
        }
        wandb_run.log(
            {
                "final/samples": total,
                "final/failed": failed,
                "final/accuracy": acc,
                "final/avg_rounds": avg_rounds,
                "final/invalid_outputs": invalid_outputs,
                "final/total_model_calls": total_model_calls,
                "final/fallback_frames_used": fallback_frames_used,
            }
        )
        wandb_run.finish()

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if server_proc is not None:
        stop_server(server_proc)

    print(json.dumps({"results": results, "summary_json": args.summary_json, "log_jsonl": log_jsonl}, indent=2))
    if total > 0 and (failed >= total or total_model_calls == 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
