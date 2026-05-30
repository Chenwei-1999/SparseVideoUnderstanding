"""Shared utility functions for REVISE plug-and-play evaluation scripts.

This module consolidates functions that were previously duplicated across multiple
standalone evaluation scripts (plug_and_play_nextqa_vllm.py, plug_and_play_egoschema_vllm.py,
plug_and_play_videomme_lvbench_vllm.py, plug_and_play_lvbench_hf.py, eval_nextqa_caption_vllm.py)
and the RL agent loop (verl/experimental/agent_loop/revise_agent_loop.py).
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Optional

import requests
from PIL import Image

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


# ---------------------------------------------------------------------------
# Regex constants for parsing model output tags
# ---------------------------------------------------------------------------

SUMMARIZE_RE = re.compile(r"<summarize>(.*?)</summarize>", re.DOTALL | re.IGNORECASE)
SELECT_RE = re.compile(r"<select>(.*?)</select>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

PLACEHOLDER_SET = {"...", "…", "none", "n/a", "na", "null", "unknown", "unsure", "uncertain"}

OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_LEADING_OPTION_LABEL_RE = re.compile(r"^[A-Z]\s*[.)]\s*")


# ---------------------------------------------------------------------------
# Text & parsing helpers
# ---------------------------------------------------------------------------


def collapse_ws(text: str) -> str:
    """Collapse all whitespace to single spaces and strip."""
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_tag(text: str, pattern: re.Pattern[str]) -> Optional[str]:
    """Extract the *last* match of a regex tag pattern, stripped."""
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def dedupe_preserve_order(indices: list[int]) -> list[int]:
    """De-duplicate a list of ints, preserving first-occurrence order."""
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices or []:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def parse_int_list(text: str) -> list[int]:
    """Extract all integers from *text* via regex."""
    return [int(n) for n in re.findall(r"\d+", text or "")]


def normalize_answer_letter(answer_text: str, num_choices: int) -> Optional[str]:
    """Normalize a model answer to a single uppercase letter within the allowed range."""
    allowed = set(OPTION_LABELS[: max(0, num_choices)]) or {"A", "B", "C", "D", "E"}

    candidate = (answer_text or "").strip().upper()
    if candidate in allowed:
        return candidate

    # Walk every uppercase match (word-bounded first, then unbounded) and return
    # the first one within `allowed` — re.search returns only the first match,
    # which would stop at an out-of-range letter before an in-range one.
    for pattern in (r"\b([A-Z])\b", r"([A-Z])"):
        for match in re.finditer(pattern, candidate):
            if match.group(1) in allowed:
                return match.group(1)
    return None


def is_placeholder(text: str) -> bool:
    """Detect placeholder / empty summary text."""
    t = collapse_ws(text).lower()
    if not t:
        return True
    if "..." in t or "…" in t:
        return True
    if t in PLACEHOLDER_SET:
        return True
    if re.fullmatch(r"[.·•…]+", t):
        return True
    alnum = re.findall(r"[a-z0-9]+", t)
    if len(alnum) <= 1 and len(t) <= 6:
        return True
    return False


def summary_has_ohrpu(summary_text: str) -> bool:
    """Check that summary contains P/O/H/U/R keys in the correct order."""
    if summary_text is None:
        return False
    s = collapse_ws(summary_text)
    keys = ["P", "O", "H", "U", "R"]
    positions = []
    for key in keys:
        m = re.search(rf"\b{key}\s*:\s*", s, re.IGNORECASE)
        if m is None:
            return False
        positions.append(m.start())
    return all(a < b for a, b in zip(positions, positions[1:], strict=False))


def contains_banned_example(text: str) -> bool:
    """Detect accidental copying of legacy few-shot example content."""
    t = collapse_ws(text).lower()
    if not t:
        return False
    if "george approaching a shelf" in t:
        return True
    if "george pauses" in t and "shelf" in t:
        return True
    return False


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, appending ellipsis if needed."""
    if max_chars <= 0:
        return text
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def retry_feedback_text(feedback: str, *, force_answer: bool, force_instructions: str) -> str:
    """Build the retry nudge appended after an unparseable model turn.

    The non-forced branch ("respond with one of the required formats") is
    identical across callers; the forced branch differs per benchmark and is
    passed in via *force_instructions* so it stays visible at the call site.
    """
    if force_answer:
        return f"{feedback}\n{force_instructions}"
    return f"{feedback}\nPlease respond with one of the required formats."


# Forced-answer instruction strings shared by the launchers' retry nudges.
# POHR variant (NExT-QA): reminds the model of the P/O/H/U/R summary order.
FORCE_ANSWER_INSTRUCTIONS_POHR = (
    "Output ONLY <think>...</think> then <answer>LETTER</answer>. "
    "In <summarize>, include P/O/H/U/R in that exact order. "
    "In <answer>, LETTER must be a single option letter (e.g., A/B/C/D/E)."
)
# Simple variant (Video-MME / LVBench-HF): no summarize reminder.
FORCE_ANSWER_INSTRUCTIONS_SIMPLE = (
    "You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."
)


def format_mc_question(question: str, choices: list[str]) -> str:
    """Format a multiple-choice question with a REVISE answer-format reminder.

    Shared by the NExT-QA and EgoSchema launchers, which had byte-identical
    copies. Distinct from :func:`format_question_block` (no trailing protocol
    reminder) — kept separate rather than overloaded with a flag, since the two
    call sites genuinely want different trailing text.
    """
    lines = [f"Question: {question.strip()}", "", "Options:"]
    for idx, choice in enumerate(choices):
        letter = OPTION_LABELS[idx] if idx < len(OPTION_LABELS) else str(idx)
        lines.append(f"{letter}. {choice.strip()}")
    lines.append("")
    lines.append(
        "Answer with one of the option letters. Begin every reply with <think>...</think>; "
        "on a Select round add <summarize> then <select>; on the Answer round add <answer>LETTER</answer>."
    )
    return "\n".join(lines)


def _contains_auto_map(value: Any) -> bool:
    if isinstance(value, dict):
        if "auto_map" in value:
            return True
        return any(_contains_auto_map(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_auto_map(v) for v in value)
    return False


@functools.lru_cache(maxsize=16)
def should_trust_remote_code(model_path: str) -> bool:
    """Return whether a local HF snapshot needs custom model code enabled."""
    override = os.getenv("REVISE_TRUST_REMOTE_CODE", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    config_path = os.path.join(str(model_path), "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return False
    return _contains_auto_map(config)


def configure_llava_processor(processor: Any, model_config: Any | None = None) -> None:
    """Fill processor attributes needed by Transformers' local LLaVA processor (in place)."""
    if getattr(processor, "patch_size", None) is None:
        patch_size = None
        vision_config = getattr(model_config, "vision_config", None)
        if vision_config is not None:
            patch_size = getattr(vision_config, "patch_size", None)
        processor.patch_size = patch_size or 14
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = (
            getattr(model_config, "vision_feature_select_strategy", None) or "default"
        )
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = getattr(model_config, "num_additional_image_tokens", 0) or 0


def _flatten_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            chunks.append(str(item))
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in {"image", "video"}:
            chunks.append("<image>\n")
        else:
            chunks.append(str(item.get("text") or ""))
    return "".join(chunks)


def apply_processor_chat_template(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """Apply a multimodal chat template with a tokenizer fallback for local LLaVA snapshots."""
    if getattr(processor, "chat_template", None):
        return str(processor.apply_chat_template(messages, **kwargs))
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        text_messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": _flatten_chat_content(message.get("content", "")),
            }
            for message in messages
        ]
        return str(tokenizer.apply_chat_template(text_messages, **kwargs))
    return str(processor.apply_chat_template(messages, **kwargs))


def normalize_video_id(video_id: Any) -> str:
    """Normalize a video ID to a string (handles int/float from pandas)."""
    if isinstance(video_id, int):
        return str(video_id)
    if isinstance(video_id, float):
        return str(int(video_id))
    return str(video_id)


def resolve_nextqa_video_path(video_root: str, rel_path: str, video_id: Any | None = None) -> str | None:
    """Resolve NExT-QA videos across official and mirror directory layouts."""
    root = os.path.abspath(video_root)
    rel = str(rel_path or "").strip().replace("\\", os.sep).replace("/", os.sep)
    if rel and not rel.endswith(".mp4"):
        rel = f"{rel}.mp4"

    candidates: list[str] = []

    def add_candidate(path: str) -> None:
        if path and path not in candidates:
            candidates.append(path)

    if rel:
        add_candidate(os.path.join(root, rel))
        base = os.path.basename(rel)
    else:
        base = ""

    basenames = [base] if base else []
    if video_id is not None:
        video_base = normalize_video_id(video_id)
        if video_base and not video_base.endswith(".mp4"):
            video_base = f"{video_base}.mp4"
        if video_base and video_base not in basenames:
            basenames.append(video_base)

    for name in basenames:
        add_candidate(os.path.join(root, name))
        add_candidate(os.path.join(root, "NExTVideo", name))
        add_candidate(os.path.join(root, "NExTVideo", "NExTVideo", name))
        add_candidate(os.path.join(root, "videos", name))
        add_candidate(os.path.join(root, "videos", "videos", name))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def ensure_writable_hf_cache(default_home: Any | None = None) -> str:
    """Ensure Hugging Face cache env vars point to writable local storage."""
    fallback = os.environ.get("REVISE_HF_HOME") or str(
        default_home or os.path.abspath(os.path.join("data", "revise_assets", "hf_home"))
    )

    hf_home = os.environ.get("HF_HOME")
    if not hf_home or not _is_writable_dir(hf_home):
        hf_home = fallback
        os.environ["HF_HOME"] = hf_home
        os.makedirs(hf_home, exist_ok=True)

    defaults = {
        "HF_HUB_CACHE": os.path.join(hf_home, "hub"),
        "HF_DATASETS_CACHE": os.path.join(hf_home, "datasets"),
        "HF_XET_CACHE": os.path.join(hf_home, "xet"),
    }
    for key, path in defaults.items():
        current = os.environ.get(key)
        if not current or not _is_writable_dir(current):
            os.environ[key] = path
            os.makedirs(path, exist_ok=True)
    return hf_home


# ---------------------------------------------------------------------------
# Frame selection & interval math
# ---------------------------------------------------------------------------


def sample_uniform_indices(frame_count: int, n: int) -> list[int]:
    """Uniformly sample *n* indices from [0, frame_count-1]."""
    if n <= 0:
        return []
    if frame_count <= 0:
        return list(range(n))
    if n == 1:
        return [frame_count // 2]
    if frame_count == 1:
        return [0]
    return [round(i * (frame_count - 1) / (n - 1)) for i in range(n)]


def linspace(a: float, b: float, n: int) -> list[float]:
    """Pure-Python linspace (no numpy dependency)."""
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def sample_uniform_indices_inclusive(start: int, end: int, k: int) -> list[int]:
    """Uniformly sample *k* indices in [start, end] inclusive."""
    if k <= 0:
        return []
    if end < start:
        return []
    if start == end:
        return [start]
    out = [int(round(i)) for i in linspace(float(start), float(end), k)]
    out = [max(start, min(i, end)) for i in out]
    return dedupe_preserve_order(out)


def indices_to_intervals(indices: list[int]) -> list[tuple[int, int]]:
    """Convert a list of indices to inclusive [start, end] intervals."""
    if not indices:
        return []
    sorted_unique = sorted({int(i) for i in indices})
    intervals: list[tuple[int, int]] = []
    start = prev = sorted_unique[0]
    for idx in sorted_unique[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        intervals.append((start, prev))
        start = prev = idx
    intervals.append((start, prev))
    return intervals


def unseen_intervals(frame_count: int, seen_frames: list[int]) -> list[tuple[int, int]]:
    """Return unseen frame ranges as inclusive [start, end] intervals."""
    if frame_count <= 0:
        return []
    seen = sorted({int(i) for i in (seen_frames or []) if 0 <= int(i) < frame_count})
    anchors = [-1, *seen, frame_count]
    intervals: list[tuple[int, int]] = []
    for a, b in zip(anchors, anchors[1:], strict=False):
        s = a + 1
        e = b - 1
        if s <= e:
            intervals.append((s, e))
    return intervals


def in_intervals(idx: int, intervals: list[tuple[int, int]]) -> bool:
    """Check if *idx* falls within any of the given inclusive intervals."""
    for start, end in intervals:
        if start <= idx <= end:
            return True
    return False


def format_intervals(intervals: list[tuple[int, int]]) -> str:
    """Format intervals as a semicolon-separated string."""
    if not intervals:
        return "none"
    parts: list[str] = []
    for start, end in intervals:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    return "; ".join(parts)


def format_frame_list(frames: list[int]) -> str:
    """Format a frame index list for display in prompts."""
    if not frames:
        return "no frames yet"
    return ", ".join(str(int(i)) for i in frames)


def propose_candidate_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
    """Propose *k* candidate NEW frame indices using gap-filling + random fill.

    Strategy: pick midpoints of the largest gaps between already-seen frames,
    then fill remaining slots with random unseen frames.
    """
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []

    seen_sorted = sorted(i for i in seen if 0 <= i < frame_count)
    anchors = sorted(set([0, frame_count - 1, *seen_sorted]))

    candidates: list[int] = []
    gaps: list[tuple[int, int, int]] = []
    for a, b in zip(anchors, anchors[1:], strict=False):
        if b - a <= 1:
            continue
        gaps.append((b - a, a, b))
    gaps.sort(reverse=True)
    for _, a, b in gaps:
        mid = (a + b) // 2
        for d in (0, -1, 1, -2, 2, -3, 3):
            idx = mid + d
            if a < idx < b and idx not in seen and idx not in candidates:
                candidates.append(idx)
                break
        if len(candidates) >= k:
            return sorted(candidates[:k])

    need = k - len(candidates)
    if need > 0:
        remaining = [i for i in range(frame_count) if i not in seen and i not in candidates]
        if remaining:
            fill = rng.sample(remaining, k=min(need, len(remaining)))
            candidates.extend(sorted(fill))

    return sorted(candidates[:k])


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def extract_video_info(video_path: str) -> tuple[int, float]:
    """Return (total_frames, fps) using decord, with imageio fallback."""
    decord_exc: Exception | None = None
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        try:
            total_frames = int(len(vr))
            fps = float(vr.get_avg_fps() or 0.0)
            if fps <= 0:
                fps = 30.0
            return total_frames, fps
        finally:
            del vr
    except Exception as exc:
        decord_exc = exc

    try:
        import imageio

        reader = imageio.get_reader(video_path, "ffmpeg")
        try:
            meta = reader.get_meta_data() or {}
            fps = float(meta.get("fps") or 30.0)
            if fps <= 0:
                fps = 30.0
            try:
                total_frames = int(reader.count_frames())
            except Exception:
                nframes = meta.get("nframes", 0)
                try:
                    total_frames = int(nframes)
                except Exception:
                    total_frames = 0
            if total_frames <= 0:
                duration = float(meta.get("duration") or 0.0)
                total_frames = int(duration * fps) if duration > 0 else 0
            return total_frames, fps
        finally:
            reader.close()
    except Exception as exc:
        raise exc from decord_exc


def timeline_len_1fps(total_frames: int, fps: float) -> int:
    """Convert (total_frames, fps) to a 1fps timeline length (seconds)."""
    if total_frames <= 0:
        return 0
    duration_s = total_frames / max(1e-6, fps)
    return max(1, int(math.ceil(duration_s)))


def timeline_to_frame_idx(timeline_idx: int, fps: float, total_frames: int) -> int:
    """Map a 1fps timeline index (seconds) to a raw frame index."""
    if total_frames <= 0:
        return 0
    t = max(0.0, float(timeline_idx))
    idx = int(t * fps)
    return max(0, min(idx, total_frames - 1))


def extract_frames_1fps(video_path: str, timeline_indices: list[int]) -> list[Image.Image]:
    """Extract frames mapped from a 1fps timeline to actual frame indices."""
    if not timeline_indices:
        return []
    import decord

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    try:
        total_frames = int(len(vr))
        fps = float(vr.get_avg_fps() or 0.0)
        if fps <= 0:
            fps = 30.0

        frame_indices = [timeline_to_frame_idx(i, fps, total_frames) for i in timeline_indices]
        try:
            frames = vr.get_batch(frame_indices).asnumpy()
            return [Image.fromarray(frame) for frame in frames]
        except Exception:
            return [Image.fromarray(vr[idx].asnumpy()) for idx in frame_indices]
    finally:
        del vr


def extract_frames(video_path: str, frame_indices: list[int]) -> list[Image.Image]:
    """Extract frames by raw index using decord (with imageio fallback)."""
    if not frame_indices:
        return []
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        try:
            frames = vr.get_batch(frame_indices).asnumpy()
            return [Image.fromarray(frame) for frame in frames]
        finally:
            del vr
    except Exception:
        pass
    import imageio

    reader = imageio.get_reader(video_path, "ffmpeg")
    try:
        return [Image.fromarray(reader.get_data(idx)) for idx in frame_indices]
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def b64_jpeg(img: Image.Image, *, max_edge: int = 0, quality: int = 90) -> str:
    """Base64-encode an image as JPEG, optionally resizing to *max_edge*.

    Args:
        img: PIL Image to encode.
        max_edge: If > 0, thumbnail the image so its longest edge is at most this.
        quality: JPEG quality (1-100).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max_edge > 0:
        w, h = img.size
        if max(w, h) > max_edge:
            img = img.copy()
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Time-reference parsing (LVBench)
# ---------------------------------------------------------------------------


def parse_time_to_seconds(text: str) -> Optional[float]:
    """Parse 'MM:SS' or 'HH:MM:SS' to seconds."""
    raw = collapse_ws(text)
    if not raw:
        return None
    parts = [p for p in raw.split(":") if p]
    if len(parts) == 2:
        try:
            mm = int(parts[0])
            ss = float(parts[1])
            return max(0.0, mm * 60.0 + ss)
        except Exception:
            return None
    if len(parts) == 3:
        try:
            hh = int(parts[0])
            mm = int(parts[1])
            ss = float(parts[2])
            return max(0.0, hh * 3600.0 + mm * 60.0 + ss)
        except Exception:
            return None
    return None


def parse_time_reference_range(time_reference: str, timeline_len: int) -> Optional[tuple[int, int]]:
    """Parse LVBench time_reference (e.g. '04:19-08:41') into a (start, end) range on the 1fps timeline."""
    tr = collapse_ws(time_reference)
    if not tr or tr.lower() in {"n/a", "na", "none"}:
        return None
    if "-" not in tr:
        return None
    left, right = (s.strip() for s in tr.split("-", 1))
    start_s = parse_time_to_seconds(left)
    end_s = parse_time_to_seconds(right)
    if start_s is None or end_s is None:
        return None
    if timeline_len <= 0:
        return None

    start = int(math.floor(start_s))
    end = int(math.ceil(end_s))
    if end < start:
        start, end = end, start
    start = max(0, min(start, timeline_len - 1))
    end = max(0, min(end, timeline_len - 1))
    if end < start:
        return None
    return start, end


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------


def format_question_block(question: str, options: list[str]) -> str:
    """Format a question with labeled options (A, B, C, ...)."""
    q = str(question).strip()
    lines = ["Question: " + q, "Options:"]
    for i, opt in enumerate(options):
        prefix = OPTION_LABELS[i] if i < len(OPTION_LABELS) else str(i)
        lines.append(f"{prefix}. {str(opt).strip()}")
    return "\n".join(lines)


def clean_videoespresso_option(option: Any) -> str:
    """Match the official VideoEspresso option cleanup before prompting."""
    text = str(option or "").strip()
    return text.split("):", 1)[-1].strip()


def format_videoespresso_question_block(
    question: str,
    options: list[str],
    *,
    task: str = "",
    evidence: str = "",
    with_evidence: bool = False,
    revise_answer_tags: bool = False,
) -> str:
    """Format VideoEspresso close-ended prompts in the official evaluation style.

    The official evaluator asks the model to finish the row's task, then passes
    the question, optional evidence, four labeled options, and asks for only an
    option letter.  REVISE runs keep that benchmark query intact and append the
    protocol-specific answer tag requirement.
    """
    task_text = str(task or "video question answering").strip()
    question_text = str(question or "").strip()

    option_lines: list[str] = []
    for idx, opt in enumerate(options):
        label = OPTION_LABELS[idx] if idx < len(OPTION_LABELS) else str(idx)
        option_lines.append(f"({label}) {clean_videoespresso_option(opt)}")
    options_prompt = "\n".join(option_lines)

    if with_evidence and str(evidence or "").strip():
        query = (
            f"Please finish the {task_text} task. Question: {question_text}. "
            f"Your inference evidence is {str(evidence).strip()}. "
            f"You have the following options: {options_prompt}. "
            "Select the answer and only give the option letters."
        )
    else:
        query = (
            f"Please finish the {task_text} task. Question: {question_text}. "
            f"You have the following options: {options_prompt}. "
            "Select the answer and only give the option letters."
        )

    if revise_answer_tags:
        allowed = ", ".join(OPTION_LABELS[i] for i in range(min(len(options), len(OPTION_LABELS))))
        query += (
            "\nFor this REVISE run, output <summarize>...</summarize> then "
            f"<answer>LETTER</answer> when answering. LETTER must be one of: {allowed}."
        )
    return query


def format_videomme_question_block(
    question: str,
    options: list[str],
    *,
    subtitles: str = "",
    with_subtitles: bool = False,
) -> str:
    """Format Video-MME prompts following the official multiple-choice template."""
    question_text = str(question or "").strip()
    option_lines: list[str] = []
    for idx, opt in enumerate(options):
        label = OPTION_LABELS[idx] if idx < len(OPTION_LABELS) else str(idx)
        text = _LEADING_OPTION_LABEL_RE.sub("", str(opt or "").strip()).strip()
        option_lines.append(f"{label}. {text}")
    question_block = "\n".join([question_text, *option_lines]).strip()

    if with_subtitles:
        subtitle_text = str(subtitles or "No subtitles available").strip()
        return (
            "This video's subtitles are listed below:\n"
            f"{subtitle_text}\n"
            "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"{question_block}\n"
            "The best answer is:"
        )
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        f"{question_block}\n"
        "The best answer is:"
    )


# ---------------------------------------------------------------------------
# Logging & monitoring
# ---------------------------------------------------------------------------


def maybe_log_jsonl(path: Optional[str], obj: dict[str, Any]) -> None:
    """Append a JSON object to a JSONL file (no-op if *path* is falsy)."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def maybe_init_wandb(args: argparse.Namespace, run_config: dict[str, Any]) -> Any:
    """Initialize a wandb run if ``--use-wandb`` is set and credentials are available.

    Expects *args* to have: ``use_wandb``, ``wandb_project``, ``wandb_entity``,
    ``wandb_name``, ``wandb_group``, ``wandb_tags``, ``wandb_mode``.
    """
    if not getattr(args, "use_wandb", False) or wandb is None:
        return None

    def _has_wandb_credentials() -> bool:
        if os.getenv("WANDB_API_KEY"):
            return True
        if os.getenv("WANDB_IDENTITY_TOKEN_FILE"):
            return True
        try:
            from wandb.sdk.lib.wbauth import wbnetrc  # type: ignore

            base_url = os.getenv("WANDB_BASE_URL") or "https://api.wandb.ai"
            return bool(wbnetrc.read_netrc_auth(host=base_url))
        except Exception:
            return False

    mode = getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE")
    if not mode:
        mode = "online" if _has_wandb_credentials() else "offline"
    if str(mode).lower() == "online" and not _has_wandb_credentials():
        mode = "offline"

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        mode=mode,
        config=run_config,
        reinit=True,
    )


def wandb_log(run: Any, metrics: dict[str, Any], step: int) -> None:
    """Log metrics to wandb (no-op if *run* is None)."""
    if run is None or wandb is None:
        return
    wandb.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Stable sample ID
# ---------------------------------------------------------------------------


def stable_sample_id(*, keys: dict[str, Any]) -> str:
    """SHA1 hash of arbitrary key-value metadata for deterministic sample IDs."""
    return hashlib.sha1(json.dumps(keys, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def stable_sample_id_nextqa(video_id: str, question: str, choices: list[str], answer_idx: int) -> str:
    """NExT-QA specific sample ID."""
    return stable_sample_id(
        keys={
            "video_id": str(video_id),
            "question": str(question),
            "choices": [str(c) for c in (choices or [])],
            "answer_idx": int(answer_idx),
        }
    )


def stable_sample_id_dataset(dataset: str, video_key: str, uid: str) -> str:
    """Video-MME / LVBench style sample ID."""
    return stable_sample_id(keys={"dataset": str(dataset), "video": str(video_key), "uid": str(uid)})


# ---------------------------------------------------------------------------
# LVBench question parsing
# ---------------------------------------------------------------------------


_OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def parse_options_from_lvbench_question(question: str) -> tuple[str, list[str]]:
    """Parse LVBench question format with ``(A)`` prefix options."""
    text = str(question or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", []

    q_lines: list[str] = []
    opts: dict[str, str] = {}
    for ln in lines:
        m = _OPT_RE.match(ln)
        if m:
            opts[m.group(1).upper()] = m.group(2).strip()
        else:
            q_lines.append(ln)

    q_text = " ".join(q_lines).strip()
    if not opts:
        return q_text, []
    letters_sorted = sorted(opts.keys())
    options = [opts[k] for k in letters_sorted]
    return q_text, options


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def shard_by_video(samples: list[Any], num_shards: int, shard_idx: int, *, video_key_attr: str = "video_key") -> list:
    """Shard a sample list by video key (deterministic round-robin)."""
    if num_shards <= 1:
        return samples
    if not (0 <= shard_idx < num_shards):
        raise ValueError(f"--shard-idx must be in [0, {num_shards}) (got {shard_idx})")

    by_video: dict[str, list] = {}
    for s in samples:
        vk = getattr(s, video_key_attr)
        by_video.setdefault(vk, []).append(s)
    video_keys = sorted(by_video.keys())
    my_videos = {vk for i, vk in enumerate(video_keys) if (i % num_shards) == shard_idx}

    out: list = []
    for vk in video_keys:
        if vk not in my_videos:
            continue
        out.extend(by_video[vk])
    return out


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------


def pick_free_port() -> int:
    """Pick a free TCP port on localhost."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def port_is_open(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            return sock.connect_ex((host, port)) == 0
        except Exception:
            return False


def wait_port(host: str, port: int, timeout_s: int = 300) -> None:
    """Wait until *host:port* accepts TCP connections."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {host}:{port} to accept connections after {timeout_s}s")


def get_api_headers(api_key: str | None = None) -> dict[str, str]:
    """Build auth headers for OpenAI-compatible APIs from args or environment."""
    token = (
        api_key
        or os.getenv("REVISE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("VLLM_API_KEY")
        or ""
    ).strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def resolve_base_url(base_url: str | None = None, host: str = "127.0.0.1", port: int = 8000) -> str:
    """Resolve an OpenAI-compatible API base URL."""
    if base_url:
        return str(base_url).rstrip("/")
    return f"http://{host}:{port}"


def wait_for_server(host: str, port: int, timeout_s: int, *, base_url: str | None = None) -> None:
    """Wait until the vLLM OpenAI server is actually ready (polls ``/v1/models``).

    vLLM can open the TCP port before the model is fully initialized; issuing a chat
    request too early can return transient HTTP 400/503 errors.
    """
    base_url = resolve_base_url(base_url, host, port)
    headers = get_api_headers()
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=1.0)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"vLLM server did not become ready at {host}:{port} within {timeout_s}s")


def get_model_id(base_url: str, timeout: int = 30, model_id: str | None = None) -> str:
    """Fetch the model ID from a running vLLM/OpenAI-compatible server."""
    explicit_model_id = (
        model_id
        or os.getenv("REVISE_MODEL_ID")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("OPENAI_API_MODEL")
        or ""
    ).strip()
    if explicit_model_id:
        return explicit_model_id

    resp = requests.get(f"{base_url}/v1/models", headers=get_api_headers(), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {base_url}/v1/models")
    return models[0]["id"]


def build_vllm_serve_command(
    args: argparse.Namespace,
    *,
    image_limit: int,
    cuda_visible_default: str = "0",
) -> tuple[list[str], dict[str, str]]:
    """Build the ``vllm serve`` argv and process environment for a launcher.

    This is the shared specification of how the plug-and-play scripts start their
    vLLM server. It is a *pure* function (no spawning, no I/O) so the spawn itself
    stays in the caller and can be mocked there; that also makes the command
    directly testable.

    The two things that legitimately differ between callers are surfaced as
    explicit arguments rather than being buried in three near-identical copies:

    * ``image_limit`` -- the ``--limit-mm-per-prompt`` image cap. NExTQA passes
      ``max(max_frames_per_round, caption_gen_max_frames)`` (it also samples
      caption frames); the others pass ``max_frames_per_round``.
    * ``cuda_visible_default`` -- the ``CUDA_VISIBLE_DEVICES`` fallback used only
      when the variable is not already set in the environment.
    """
    env = os.environ.copy()
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", cuda_visible_default)

    # Prefer a vLLM binary from the active Python environment.
    py_bin = os.path.dirname(sys.executable)
    if py_bin:
        env["PATH"] = py_bin + os.pathsep + env.get("PATH", "")
    vllm_bin = shutil.which("vllm", path=env.get("PATH"))

    cmd = [
        vllm_bin or "vllm",
        "serve",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--load-format",
        "auto",
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--limit-mm-per-prompt",
        json.dumps({"image": int(image_limit)}),
    ]
    if getattr(args, "model_id", None):
        cmd += ["--served-model-name", str(args.model_id)]
    if should_trust_remote_code(str(args.model_path)):
        cmd += ["--trust-remote-code"]
    return cmd, env


def open_server_log_streams(server_log: str | None) -> tuple[Any, Any]:
    """Resolve (stdout, stderr) targets for a launched server.

    Returns ``DEVNULL`` for both when no log path is given, otherwise an appended
    log file (creating its parent directory) used for both streams.
    """
    if not server_log:
        return subprocess.DEVNULL, subprocess.DEVNULL
    log_dir = os.path.dirname(server_log)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_f = open(server_log, "a", encoding="utf-8")
    return log_f, log_f


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate a server subprocess."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def suffix_path(path: str, suffix: str) -> str:
    """Add a suffix before the file extension (e.g. ``'log.jsonl'`` -> ``'log.shard0of4.jsonl'``)."""
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completion (shared by the plug-and-play launchers)
# ---------------------------------------------------------------------------


def build_chat_content(
    user_text: str,
    images: list[Image.Image],
    *,
    max_edge: int = 0,
    quality: int = 90,
) -> list[dict[str, Any]]:
    """Interleave text segments with base64 images at ``<image>`` placeholders.

    The user text carries ``<image>`` markers produced by the prompt builder;
    each marker is replaced by the next image so the model sees frames in
    position. Any images beyond the number of placeholders are appended at the
    end (defensive: keeps every frame visible even if the text drifts).

    *max_edge* / *quality* tune the JPEG encoding so high-resolution callers can
    cap payload size; the default (no resize, q=90) preserves the original
    NExT-QA behavior.
    """
    parts = user_text.split("<image>") if images else [user_text]
    content: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < len(parts) - 1 and i < len(images):
            b64 = b64_jpeg(images[i], max_edge=max_edge, quality=quality)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    for j in range(len(parts) - 1, len(images)):
        b64 = b64_jpeg(images[j], max_edge=max_edge, quality=quality)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def chat_once(
    base_url: str,
    model_id: str,
    system_prompt: str,
    user_text: str,
    images: list[Image.Image],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
    *,
    max_edge: int = 0,
    quality: int = 90,
) -> str:
    """Issue one OpenAI-compatible chat completion with interleaved frames.

    On an HTTP error the response body is extracted and truncated into the
    raised ``RuntimeError`` so launcher logs show *why* vLLM rejected a request
    (a bare ``raise_for_status`` would hide the server's message).
    """
    content = build_chat_content(user_text, images, max_edge=max_edge, quality=quality)
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    headers.update(get_api_headers())
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers, timeout=timeout_s)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body = (resp.text or "")[:2000]
        raise RuntimeError(f"vLLM HTTP {resp.status_code}: {body}") from exc
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


def start_vllm_server(
    args: "argparse.Namespace",
    *,
    image_limit: int,
    cuda_visible_default: str = "0",
) -> "subprocess.Popen[str]":
    """Spawn a ``vllm serve`` subprocess from a launcher's parsed args.

    The pure command/env construction lives in :func:`build_vllm_serve_command`;
    this wrapper only performs the side-effecting spawn and log-stream wiring so
    the four launchers share one definition. The two genuinely per-caller knobs
    (*image_limit*, *cuda_visible_default*) stay explicit at the call site.
    """
    cmd, env = build_vllm_serve_command(args, image_limit=image_limit, cuda_visible_default=cuda_visible_default)
    server_stdout, server_stderr = open_server_log_streams(getattr(args, "server_log", None))
    # Pass env= so the child vLLM process inherits the CUDA_VISIBLE_DEVICES /
    # PATH that build_vllm_serve_command set up (all three original launchers
    # did this; dropping it would break GPU selection and vllm-binary lookup).
    return subprocess.Popen(cmd, env=env, stdout=server_stdout, stderr=server_stderr)
