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
from revise import pnp_engine
from revise.pnp_protocols import LoopConfig, RunStats
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


class _BaseLongVideoDataset:
    def __init__(
        self,
        *,
        split: str,
        video_cache_dir: str,
        yt_dlp_timeout_s: int,
        videomme_use_official_prompt: bool = True,
    ) -> None:
        self.split = split
        self.video_cache_dir = video_cache_dir
        self.yt_dlp_timeout_s = int(yt_dlp_timeout_s)
        self.videomme_use_official_prompt = bool(videomme_use_official_prompt)
        self.think_present = 0
        self.missing_summary = 0
        self._current_sample: Optional[MCVideoSample] = None
        self._video_path_cache: dict[str, str] = {}
        self._frame_count_cache: dict[str, int] = {}

    def _cache_path(self, sample: MCVideoSample) -> Path:
        return Path(self.video_cache_dir) / sample.dataset / sample.video_key

    def _ensure_video_path(self, sample: MCVideoSample) -> str:
        cached = self._video_path_cache.get(sample.sample_id)
        if cached is not None:
            return cached
        video_path = self._cache_path(sample)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        failed_marker = str(video_path) + ".failed"
        video_ok = video_path.exists() and video_path.stat().st_size > 0
        if video_ok and os.path.exists(failed_marker):
            try:
                os.remove(failed_marker)
            except Exception:
                pass
        if not video_ok and os.path.exists(failed_marker):
            raise RuntimeError("download_failed_cached")
        if not video_ok:
            try:
                _download_youtube(
                    sample.video_url,
                    str(video_path),
                    py_bin=sys.executable,
                    timeout_s=self.yt_dlp_timeout_s,
                )
            except Exception as e:
                try:
                    with open(failed_marker, "w", encoding="utf-8") as f:
                        f.write(f"download_failed: {type(e).__name__}: {str(e)}\n")
                except Exception:
                    pass
                raise RuntimeError(f"download_failed: {type(e).__name__}: {str(e)[:400]}") from e
        out = str(video_path)
        self._video_path_cache[sample.sample_id] = out
        return out

    def video_path(self, sample: MCVideoSample) -> str:
        return self._ensure_video_path(sample)

    def frame_count(self, sample: MCVideoSample) -> int:
        cached = self._frame_count_cache.get(sample.sample_id)
        if cached is not None:
            return cached
        video_path = self._ensure_video_path(sample)
        total_frames, fps = extract_video_info(video_path)
        timeline_len = timeline_len_1fps(total_frames, fps)
        if timeline_len <= 0:
            raise RuntimeError("invalid_video_timeline")
        self._frame_count_cache[sample.sample_id] = timeline_len
        return timeline_len

    def video_id(self, sample: MCVideoSample) -> str:
        return sample.video_key

    def num_choices(self, sample: MCVideoSample) -> int:
        return len(sample.options)

    def normalize_answer(self, sample: MCVideoSample, answer_text: str) -> Optional[str]:
        return normalize_answer_letter(answer_text, self.num_choices(sample))

    def ground_truth_letter(self, sample: MCVideoSample) -> Optional[str]:
        return normalize_answer_letter(sample.answer_letter, self.num_choices(sample))

    def is_correct(self, sample: MCVideoSample, pred_letter: str) -> bool:
        return pred_letter == self.ground_truth_letter(sample)

    def log_fields(self, sample: MCVideoSample) -> dict[str, Any]:
        return {
            "dataset": sample.dataset,
            "split": self.split,
            "sample_id": sample.sample_id,
            "uid": sample.uid,
            "video_id": sample.video_key,
            "video_key": sample.video_key,
            "video_url": sample.video_url,
            "video_path": str(self._cache_path(sample)),
            "question": sample.question,
            "choices": sample.options,
            "answer_gt": sample.answer_letter,
        }

    def format_question(self, sample: MCVideoSample) -> str:
        self._current_sample = sample
        return format_question_block(sample.question, sample.options)

    def system_prompt(self, cfg: LoopConfig) -> str:
        return DEFAULT_SYSTEM_PROMPT_WITH_THINK.format(max_frames_per_round=cfg.max_frames_per_round)

    def build_user_text(self, **kwargs: Any) -> str:
        sample = self._current_sample
        return _build_user_text(
            question_block=kwargs["question_block"],
            summary=kwargs["summary"],
            timeline_len=kwargs["frame_count"],
            round_idx=kwargs["round_idx"],
            current_frames=kwargs["frame_indices"],
            seen_frames=kwargs["seen_frames"],
            candidate_unseen_frames=kwargs.get("candidate_unseen_frames") or [],
            use_candidate_frame_ids=bool(kwargs.get("use_candidate_frame_ids", False)),
            require_candidate_frames=bool(kwargs.get("require_candidate_frames", False)),
            time_reference=sample.time_reference if sample is not None else "",
            num_options=len(sample.options) if sample is not None else 0,
        )

    def extract_frames(self, sample: MCVideoSample, indices: list[int]) -> list[Image.Image]:
        return extract_frames_1fps(self._ensure_video_path(sample), indices)

    def sample_unseen_frames(self, frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
        if frame_count <= 0 or k <= 0:
            return []
        candidates = [i for i in range(frame_count) if i not in seen]
        if not candidates:
            return []
        return sorted(rng.sample(candidates, k=min(k, len(candidates))))

    def _active_range(self, sample: MCVideoSample, frame_count: int) -> tuple[int, int]:
        _ = sample
        return 0, max(0, int(frame_count) - 1)

    def initial_frame_indices(self, sample: MCVideoSample, frame_count: int, cfg: LoopConfig) -> list[int]:
        start, end = self._active_range(sample, frame_count)
        return sample_uniform_indices_inclusive(start, end, cfg.max_frames_per_round)

    def candidate_frame_indices(
        self,
        sample: MCVideoSample,
        *,
        frame_count: int,
        seen_frames: list[int],
        k: int,
        rng: random.Random,
    ) -> list[int]:
        start, end = self._active_range(sample, frame_count)
        local_len = max(0, end - start + 1)
        if local_len <= 0:
            return []
        seen_local = {int(i - start) for i in seen_frames if start <= int(i) <= end}
        candidate_local = propose_candidate_frames(frame_count=local_len, seen=seen_local, k=k, rng=rng)
        return [int(i + start) for i in candidate_local]

    def fallback_frame_indices(self, sample: MCVideoSample, frame_count: int, k: int, cfg: LoopConfig) -> list[int]:
        _ = cfg
        start, end = self._active_range(sample, frame_count)
        return sample_uniform_indices_inclusive(start, end, k)

    def retry_feedback_text(
        self,
        reason: str,
        *,
        force_answer: bool = False,
        max_frames_per_round: int = 0,
        frame_count: int = 0,
        seen_frames: Optional[list[int]] = None,
    ) -> str:
        _ = max_frames_per_round, frame_count, seen_frames
        messages = {
            "invalid_answer_letter": "Invalid response: <answer> must be a single option letter.",
            "missing_frames_tag": (
                "Invalid response: provide either <answer>LETTER</answer> OR <select>id1,id2</select> "
                "after the <summarize>."
            ),
            "frames_out_of_range": "Invalid response: requested frames must be within candidate IDs.",
            "frames_not_in_candidates": "Invalid response: requested frames must be within candidate IDs.",
            "invalid_frames": (
                "Invalid response: provide either <answer>LETTER</answer> OR <select>id1,id2</select> "
                "after the <summarize>."
            ),
        }
        return _retry_feedback_text(messages.get(reason, reason), force_answer=force_answer)

    def load_video_captions(self, captions_dir: str, video_id: str) -> dict[int, str]:
        _ = captions_dir, video_id
        return {}

    def get_video_fps(self, video_path: str) -> float:
        _ = video_path
        return 0.0

    def caption_key_for_frame_index(self, frame_idx: int, fps: float) -> int:
        _ = fps
        return int(frame_idx)

    def parse_think(self, raw: str) -> Optional[str]:
        think = extract_tag(raw, THINK_RE)
        if think is not None:
            self.think_present += 1
            return think
        # Historical Video-MME/LVBench behavior counted missing <think> only as
        # a diagnostic; answer/select parsing still proceeded.
        return ""

    def parse_summary(self, raw: str) -> Optional[str]:
        summary = extract_tag(raw, SUMMARIZE_RE)
        if summary is None:
            self.missing_summary += 1
        return summary

    def parse_answer(self, raw: str) -> Optional[str]:
        tagged = extract_tag(raw, ANSWER_RE)
        if tagged is not None:
            return tagged
        bare = _bare_answer_after_summary(raw)
        sample = self._current_sample
        if bare is not None and sample is not None and normalize_answer_letter(bare, len(sample.options)) is not None:
            return bare
        return None

    def parse_select(self, raw: str) -> Optional[str]:
        tagged = extract_tag(raw, SELECT_RE)
        if tagged is not None:
            return tagged
        bare = _bare_answer_after_summary(raw)
        sample = self._current_sample
        if bare is not None and sample is not None and normalize_answer_letter(bare, len(sample.options)) is not None:
            return None
        tail = raw or ""
        m_end = None
        for m in re.finditer(r"</summarize>", raw or "", flags=re.IGNORECASE):
            m_end = m.end()
        if m_end is not None:
            tail = (raw or "")[m_end:]
        requested = dedupe_preserve_order(parse_int_list(tail))
        return ", ".join(str(i) for i in requested) if requested else None

    def should_commit_summary(self, summary: Optional[str], *, seen_count: int) -> bool:
        _ = seen_count
        return summary is not None

    def is_select_summary_valid(self, summary: Optional[str], *, seen_count: int) -> bool:
        _ = summary, seen_count
        return True

    def is_summary_stale(self, summary: Optional[str], *, seen_count: int) -> bool:
        _ = summary, seen_count
        return False

    def select_has_range_syntax(self, frames_text: str) -> bool:
        _ = frames_text
        return False

    def parse_select_frames(self, frames_text: str) -> list[int]:
        return dedupe_preserve_order(parse_int_list(frames_text))

    def map_candidate_frame_ids(
        self,
        requested_ids: list[int],
        candidate_frames: list[int],
    ) -> Optional[list[int]]:
        mapped: list[int] = []
        allowed = {int(x) for x in candidate_frames}
        for cid in requested_ids:
            if 1 <= cid <= len(candidate_frames):
                mapped.append(int(candidate_frames[cid - 1]))
            elif int(cid) in allowed:
                mapped.append(int(cid))
        if requested_ids and not mapped:
            return None
        return dedupe_preserve_order(mapped)

    def filter_requested_frames(
        self,
        requested_frames: list[int],
        *,
        frame_count: int,
        seen_frames: list[int],
        candidate_frames: list[int],
        require_candidate_frames: bool = False,
    ) -> tuple[list[int], Optional[str]]:
        if require_candidate_frames and candidate_frames:
            allowed = {int(i) for i in candidate_frames}
            if any(int(i) not in allowed for i in requested_frames):
                return [], "frames_not_in_candidates"
            requested_frames = [i for i in requested_frames if int(i) in allowed]
        return (
            [int(i) for i in requested_frames if 0 <= int(i) < frame_count and int(i) not in seen_frames],
            None,
        )

    def initial_summary(self, cfg: LoopConfig) -> str:
        _ = cfg
        return (
            "P: the agent has not seen any frames yet; "
            "O: no reliable observation yet; "
            "H: my belief will be updated based on what is observed; "
            "U: key detail is still unclear; "
            "R: need evidence from frames"
        )

    def final_round_instruction(self, cfg: LoopConfig) -> Optional[str]:
        _ = cfg
        return (
            "This is the final round. You MUST answer now using <think>...</think> then "
            "<think>...</think> then <answer>LETTER</answer>."
        )

    def final_answer_instruction(self, cfg: LoopConfig) -> Optional[str]:
        _ = cfg
        return "You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."

    def forced_answer_request(
        self,
        sample: MCVideoSample,
        *,
        question_block: str,
        frame_count: int,
        max_rounds: int,
        system_prompt: str,
        last_user_text: str,
        last_images: list[Any],
    ) -> tuple[str, str, list[Any]]:
        _ = sample, frame_count, max_rounds, system_prompt, last_user_text
        user_text = (
            f"{question_block}\n"
            "You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."
        )
        return FINAL_ANSWER_SYSTEM_PROMPT, user_text, last_images

    def should_terminate_on_invalid_summary(self, cfg: LoopConfig) -> bool:
        _ = cfg
        return False

    def should_fail_on_empty_images(self, cfg: LoopConfig) -> bool:
        _ = cfg
        return False

    def should_count_exhausted_invalid_as_retry(self, reason: str) -> bool:
        return reason in {
            "invalid_answer_letter",
            "missing_frames_tag",
            "frames_not_in_candidates",
            "invalid_frames",
        }

    def should_clear_frame_plan_on_exhausted_invalid(self, reason: str) -> bool:
        return reason in {
            "invalid_answer_letter",
            "missing_frames_tag",
            "frames_not_in_candidates",
            "invalid_frames",
        }

    def should_retry_invalid_output(self, reason: str) -> bool:
        return reason != "frames_out_of_range"


class VideoMMEDataset(_BaseLongVideoDataset):
    def format_question(self, sample: MCVideoSample) -> str:
        self._current_sample = sample
        if self.videomme_use_official_prompt:
            return format_videomme_question_block(sample.question, sample.options)
        return format_question_block(sample.question, sample.options)


class LVBenchDataset(_BaseLongVideoDataset):
    def _active_range(self, sample: MCVideoSample, frame_count: int) -> tuple[int, int]:
        if sample.time_reference:
            parsed = parse_time_reference_range(sample.time_reference, frame_count)
            if parsed is not None:
                return parsed
        return 0, max(0, int(frame_count) - 1)


class VllmHttpBackend:
    def __init__(self, restart_server: Optional[Any] = None) -> None:
        self.restart_server = restart_server

    def chat(self, **kwargs: Any) -> str:
        try:
            return _chat_once(**kwargs)
        except Exception:
            if self.restart_server is None:
                raise
            self.restart_server()
            return _chat_once(**kwargs)

    def get_model_id(self, base_url: str, model_id: Optional[str] = None) -> str:
        return get_model_id(base_url, model_id=model_id)


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
    stats = RunStats()
    dataset_adapter: _BaseLongVideoDataset
    if args.dataset == "videomme":
        dataset_adapter = VideoMMEDataset(
            split=split,
            video_cache_dir=args.video_cache_dir,
            yt_dlp_timeout_s=args.yt_dlp_timeout_s,
            videomme_use_official_prompt=bool(args.videomme_use_official_prompt),
        )
    else:
        dataset_adapter = LVBenchDataset(
            split=split,
            video_cache_dir=args.video_cache_dir,
            yt_dlp_timeout_s=args.yt_dlp_timeout_s,
            videomme_use_official_prompt=bool(args.videomme_use_official_prompt),
        )
    cfg = LoopConfig(
        max_rounds=args.max_rounds,
        max_frames_per_round=args.max_frames_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        request_timeout_s=args.timeout_s,
        max_retries_per_round=args.max_retries_per_round,
        strict_actions=False,
        force_final_answer=bool(args.force_final_answer),
        use_candidate_frames=True,
        candidate_k=int(args.candidate_k),
        use_candidate_frame_ids=bool(args.use_candidate_frame_ids),
        require_candidate_frames=bool(args.require_candidate_frames),
        answer_only_final_round=False,
        observation_mode="image",
        caption_include="none",
        caption_max_chars=0,
        captions_dir=None,
        hide_seen_frames_in_prompt=False,
        log_jsonl=args.log_jsonl or None,
        seed=42 + int(args.shard_idx),
    )

    def _restart_server() -> None:
        nonlocal server_proc, model_id
        stats.total_retries += 1
        if server_proc is not None:
            stop_server(server_proc)
        server_proc = _start_vllm_server(args)
        wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
        model_id = get_model_id(base_url, model_id=args.model_id)

    backend = VllmHttpBackend(
        restart_server=_restart_server if (args.restart_server_on_failure and args.start_server) else None
    )

    start_t = time.time()

    def _process_one(sample: MCVideoSample) -> None:
        nonlocal correct, total_rounds, invalid_outputs, invalid_action_terminated, failed
        nonlocal total_model_calls, total_retries, total_effective_rounds, think_present
        nonlocal missing_summary, answered, total_frames_used_all, total_frames_used_answered

        try:
            outcome = pnp_engine.run_sample(
                sample,
                dataset=dataset_adapter,
                backend=backend,
                cfg=cfg,
                stats=stats,
                rng=rng,
                base_url=base_url,
                model_id=model_id,
                run=run,
            )
        except Exception as e:
            stats.failed += 1
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
                    "video_path": str(Path(args.video_cache_dir) / sample.dataset / sample.video_key),
                    "error": f"{type(e).__name__}: {str(e)[:400]}",
                },
            )
        else:
            if outcome.answer_letter is not None:
                frames_used = len(outcome.seen_frames)
                total_frames_used_all += frames_used
                total_frames_used_answered += frames_used
                answered += 1
                total_rounds += min(outcome.round_idx, args.max_rounds)
                total_effective_rounds += outcome.effective_rounds
                if dataset_adapter.is_correct(sample, outcome.answer_letter):
                    correct += 1
            elif args.force_final_answer:
                stats.invalid_action_terminated += 1
                total_frames_used_all += len(outcome.seen_frames)

        failed = stats.failed
        invalid_outputs = stats.invalid_outputs
        invalid_action_terminated = stats.invalid_action_terminated
        total_model_calls = stats.total_model_calls
        total_retries = stats.total_retries
        think_present = dataset_adapter.think_present
        missing_summary = dataset_adapter.missing_summary

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
