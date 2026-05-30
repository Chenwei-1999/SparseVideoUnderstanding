#!/usr/bin/env python3
"""REVISE multi-round plug-and-play evaluation for NExT-QA (vLLM backend).

Runs the REVISE question-aware sparse-video loop against a vLLM
OpenAI-compatible server. For each multiple-choice question the model iterates,
per round:

    <think> reason  ->  <summarize> P/O/H/U/R state  ->  <select> new frames
                                                       (or <answer> LETTER)

It starts from a uniform set of frames; each round it may request previously
unseen frames. The loop ends when the model answers or ``--max-rounds`` is
reached (a final answer is force-requested once the round budget is exhausted).
The same entry point also generates SFT teacher data.

Run as a CLI (see ``main``); invoked by ``run_generate_teacher_data.sh`` and
``scripts/paper_suite.py``. Shared helpers live in ``revise/pnp_utils.py`` and
prompts in ``revise/pnp_prompts.py``; the launch/sampling helpers are pinned by
``tests/test_pnp_characterization.py``.
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

import pandas as pd
import requests
from PIL import Image

# Allow direct execution via `python examples/...py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise.pnp_prompts import (
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_CAPTION_ONLY as DEFAULT_SYSTEM_PROMPT_CAPTION_ONLY,
)
from revise.pnp_utils import FORCE_ANSWER_INSTRUCTIONS_POHR as _FORCE_ANSWER_INSTRUCTIONS
from revise.pnp_utils import (
    ANSWER_RE,
    SELECT_RE,
    SUMMARIZE_RE,
    THINK_RE,
    b64_jpeg,
    contains_banned_example,
    dedupe_preserve_order,
    extract_frames,
    extract_tag,
    format_frame_list,
    format_intervals,
    get_api_headers,
    get_model_id,
    in_intervals,
    indices_to_intervals,
    is_placeholder,
    maybe_init_wandb,
    maybe_log_jsonl,
    normalize_answer_letter,
    normalize_video_id,
    parse_int_list,
    propose_candidate_frames,
    resolve_nextqa_video_path,
    resolve_base_url,
    sample_uniform_indices,
    stable_sample_id_nextqa,
    build_vllm_serve_command,
    open_server_log_streams,
    retry_feedback_text,
    stop_server,
    summary_has_ohrpu,
    truncate_text,
    unseen_intervals,
    wait_port,
    wait_for_server,
    wandb_log,
)

_CAPTION_CACHE: dict[str, dict[int, str]] = {}
_FPS_CACHE: dict[str, float] = {}


def _load_video_captions(captions_dir: str, video_id: str) -> dict[int, str]:
    cached = _CAPTION_CACHE.get(video_id)
    if cached is not None:
        return cached
    path = os.path.join(captions_dir, f"{video_id}_cap.json")
    if not os.path.exists(path):
        _CAPTION_CACHE[video_id] = {}
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        _CAPTION_CACHE[video_id] = {}
        return {}
    captions: dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if isinstance(v, str):
                captions[idx] = v.strip()
    _CAPTION_CACHE[video_id] = captions
    return captions


def _get_video_fps(video_path: str) -> float:
    cached = _FPS_CACHE.get(video_path)
    if cached is not None:
        return cached
    fps = 0.0
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 0.0
    _FPS_CACHE[video_path] = fps
    return fps


def _caption_key_for_frame_index(frame_idx: int, fps: float) -> int:
    if fps and fps > 0:
        return max(0, int(frame_idx / fps))
    return int(frame_idx)


def _frames_has_range_syntax(frames_text: str) -> bool:
    if not frames_text:
        return False
    # Common failure mode: model copies allowed ranges like "4-182" into <select>.
    return bool(re.search(r"\d+\s*[-–—]\s*\d+", frames_text))


def _summary_has_stale_boilerplate(summary_text: str, *, seen_count: int) -> bool:
    if not summary_text or seen_count <= 0:
        return False
    s = re.sub(r"\s+", " ", str(summary_text)).strip().lower()
    if "has not seen any frames yet" in s:
        return True
    if re.search(r"\bhas not seen any (frame|frames|caption|captions) yet\b", s):
        return True
    if "no frames yet" in s:
        return True
    if "no captions yet" in s:
        return True
    return False


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


def _build_user_text(
    question_block: str,
    summary: str,
    frame_count: int,
    round_idx: int,
    frame_indices: list[int],
    seen_frames: list[int],
    *,
    render_images: bool = True,
    hide_seen_frames: bool = False,
    candidate_unseen_frames: Optional[list[int]] = None,
    use_candidate_frame_ids: bool = False,
    require_candidate_frames: bool = False,
    shown_frame_captions: Optional[list[str]] = None,
    candidate_id_captions: Optional[list[str]] = None,
    shown_frame_ts: Optional[list[int]] = None,
    candidate_id_ts: Optional[list[int]] = None,
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
        if candidate_id_captions:
            lines.append("Captions for candidate unseen frame IDs (1fps, may be noisy):")
            for cid, cap in enumerate(candidate_id_captions, start=1):
                if candidate_id_ts and (cid - 1) < len(candidate_id_ts):
                    lines.append(f"ID {cid} (t≈{int(candidate_id_ts[cid - 1])}s): {cap}")
                else:
                    lines.append(f"ID {cid}: {cap}")
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
            lines.append(
                prefix + f"{format_intervals(indices_to_intervals(candidate_unseen_frames))}"
            )
            if require_candidate_frames:
                lines.append("In <select>, output ONLY indices within the Candidate unseen frame ranges above.")
    lines.extend(["Current summary:", f"<summarize>{summary}</summarize>"])

    if shown_frame_captions:
        lines.append(
            "Captions for shown frames (1fps, may be noisy):"
            if render_images
            else "Captions shown in this round (1fps, index≈seconds; may be noisy):"
        )
        if not render_images:
            for idx, cap in zip(frame_indices, shown_frame_captions, strict=False):
                lines.append(f"{int(idx)}s: {cap}")
        elif shown_frame_ts:
            for ts, cap in zip(shown_frame_ts, shown_frame_captions, strict=False):
                lines.append(f"{int(ts)}s: {cap}")
        else:
            use_labels = hide_seen_frames or (candidate_unseen_frames and use_candidate_frame_ids)
            if use_labels:
                labels = [chr(ord("A") + i) for i in range(len(frame_indices))]
                for label, cap in zip(labels, shown_frame_captions, strict=False):
                    lines.append(f"{label}: {cap}")
            else:
                for idx, cap in zip(frame_indices, shown_frame_captions, strict=False):
                    lines.append(f"{idx}: {cap}")

    if render_images:
        lines.append("Frames shown in this round:")
        if shown_frame_ts:
            for ts in shown_frame_ts[: len(frame_indices)]:
                lines.append(f"Shown frame at t≈{int(ts)}s <image>")
        elif hide_seen_frames or (candidate_unseen_frames and use_candidate_frame_ids):
            # Avoid leaking/copying raw frame indices when the action space is candidate IDs.
            for i, _ in enumerate(frame_indices):
                label = chr(ord("A") + i)
                lines.append(f"Shown frame {label} <image>")
        else:
            for idx in frame_indices:
                lines.append(f"Frame {idx} <image>")
    else:
        # Caption-only / text-only mode: captions above are the actual observation;
        # avoid redundant "Shown frame A/B/..." lines unless no captions were provided.
        if not shown_frame_captions:
            lines.append("Caption indices shown in this round:")
            if hide_seen_frames or (candidate_unseen_frames and use_candidate_frame_ids):
                for i, _ in enumerate(frame_indices):
                    label = chr(ord("A") + i)
                    lines.append(f"Shown {label}")
            else:
                for idx in frame_indices:
                    lines.append(str(idx))
    return "\n".join(lines)


@dataclass
class NextQASample:
    sample_id: str
    qid: str
    video_id: str
    video_path: str
    question: str
    choices: list[str]
    answer_idx: int
    frame_count: int


def _load_nextqa_samples(
    csv_path: str,
    map_json: str,
    video_root: str,
    max_samples: int,
    seed: int,
) -> list[NextQASample]:
    with open(map_json, "r", encoding="utf-8") as f:
        video_map = json.load(f)
    video_map = {str(k): v for k, v in video_map.items()}

    df = pd.read_csv(csv_path)
    if max_samples > 0:
        # Deterministic shuffle, but keep picking until we have enough valid samples.
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    samples: list[NextQASample] = []
    for _, row in df.iterrows():
        video_id = normalize_video_id(row["video"])
        rel = video_map.get(video_id)
        if rel is None:
            continue
        video_path = resolve_nextqa_video_path(video_root, str(rel), video_id)
        if video_path is None:
            continue
        choices = [str(row[f"a{i}"]) for i in range(5)]
        answer_idx = int(row.get("answer", 0))
        samples.append(
            NextQASample(
                sample_id=stable_sample_id_nextqa(video_id, str(row.get("question", "")), choices, answer_idx),
                qid=str(row.get("qid", "")),
                video_id=video_id,
                video_path=video_path,
                question=str(row.get("question", "")),
                choices=choices,
                answer_idx=answer_idx,
                frame_count=int(row.get("frame_count", 0)),
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def _sample_unseen_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []
    candidates = [i for i in range(frame_count) if i not in seen]
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates
    return sorted(rng.sample(candidates, k=k))


def _retry_feedback_text(feedback: str, *, force_answer: bool = False) -> str:
    return retry_feedback_text(
        feedback,
        force_answer=force_answer,
        force_instructions=_FORCE_ANSWER_INSTRUCTIONS,
    )


def _load_progress_from_log(
    log_path: str,
    max_rounds: int,
    default_num_choices: int = 5,
) -> tuple[int, int, int]:
    if not log_path or not os.path.exists(log_path):
        return 0, 0, 0

    completed = 0
    correct = 0
    total_rounds = 0

    in_sample = False
    current_gt_idx = -1
    current_num_choices = default_num_choices
    current_answer: Optional[str] = None
    current_answer_round: Optional[int] = None

    def _finalize_sample() -> None:
        nonlocal completed, correct, total_rounds
        nonlocal in_sample, current_gt_idx, current_num_choices, current_answer, current_answer_round
        if not in_sample:
            return
        if current_answer is None or current_answer_round is None:
            return
        completed += 1
        pred_idx = ord(current_answer) - ord("A")
        if current_gt_idx >= 0 and pred_idx == current_gt_idx:
            correct += 1
        total_rounds += min(current_answer_round, max_rounds)

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            forced = bool(obj.get("forced_answer", False))
            raw = str(obj.get("raw_output", ""))

            try:
                round_idx = int(obj.get("round_idx", 0))
            except Exception:
                round_idx = 0

            # New sample boundary: every sample starts from round_idx=1 (forced answer is logged as max_rounds+1).
            if not in_sample or (round_idx == 1 and not forced):
                _finalize_sample()
                in_sample = True
                try:
                    current_gt_idx = int(obj.get("ground_truth_idx", -1))
                except Exception:
                    current_gt_idx = -1
                choices = obj.get("choices", [])
                current_num_choices = len(choices) if isinstance(choices, list) and choices else default_num_choices
                current_answer = None
                current_answer_round = None

            answer = extract_tag(raw, ANSWER_RE)
            if answer and current_answer is None:
                answer_letter = normalize_answer_letter(answer, current_num_choices)
                if answer_letter is not None:
                    current_answer = answer_letter
                    # For forced answers, we log an extra request after max_rounds, but the "round budget" is max_rounds.
                    current_answer_round = min(round_idx, max_rounds)

    _finalize_sample()
    return completed, correct, total_rounds


def _count_file_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


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
    # Prefer interleaving text + images to match the "<image>" placeholders in text.
    # This is more robust across OpenAI-compatible multimodal frontends.
    content: list[dict[str, Any]] = []
    parts = user_text.split("<image>") if images else [user_text]
    if images and (len(parts) - 1) == len(images):
        for i, img in enumerate(images):
            if parts[i]:
                content.append({"type": "text", "text": parts[i]})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg(img)}"},
                }
            )
        if parts[-1]:
            content.append({"type": "text", "text": parts[-1]})
    else:
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg(img)}"},
                }
            )
        content.append({"type": "text", "text": user_text})

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

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=get_api_headers(),
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _start_vllm_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    # NExTQA also samples caption-generation frames, so the per-prompt image cap
    # is the max of both frame budgets. The default GPU set is all four cards.
    image_limit = max(
        int(getattr(args, "max_frames_per_round", 1)),
        int(getattr(args, "caption_gen_max_frames", 1)),
    )
    cmd, env = build_vllm_serve_command(
        args, image_limit=image_limit, cuda_visible_default="0,1,2,3"
    )
    server_stdout, server_stderr = open_server_log_streams(args.server_log)
    return subprocess.Popen(cmd, env=env, stdout=server_stdout, stderr=server_stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF model id or local snapshot path")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--map-json", required=True)
    parser.add_argument("--csv", required=True, help="NExT-QA CSV (e.g., val.csv)")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1, help="Shard the dataset for data-parallel evaluation.")
    parser.add_argument("--shard-idx", type=int, default=0, help="Shard index in [0, num_shards).")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument(
        "--max-frames-per-round",
        "--max-frames",
        type=int,
        default=5,
        dest="max_frames_per_round",
        help="Max frames to show/request per round.",
    )
    parser.add_argument(
        "--use-candidate-frames",
        action="store_true",
        help="Include a small list of candidate unseen frame indices in the prompt to help frame selection.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Number of candidate unseen frames to propose when --use-candidate-frames is set "
        "(default: max(12, max_frames_per_round*4)).",
    )
    parser.add_argument(
        "--use-candidate-frame-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --use-candidate-frames is enabled, expose candidates as IDs 1..K (not raw indices) and "
        "require <select> to output candidate IDs.",
    )
    parser.add_argument(
        "--require-candidate-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When candidate frames are provided, treat them as an allowlist: <select> must only contain indices "
        "within the candidate unseen ranges (otherwise terminate on strict actions).",
    )
    parser.add_argument(
        "--hide-seen-frames-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not print explicit seen frame indices in the prompt (reduces copying); rely on unseen ranges instead.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["image", "caption"],
        default="image",
        help="Observation source per round: 'image' shows video frames; 'caption' is caption-only (no images).",
    )
    parser.add_argument(
        "--captions-dir",
        default=None,
        help="Optional directory containing per-video caption JSON files named <video_id>_cap.json.",
    )
    parser.add_argument(
        "--caption-include",
        choices=["none", "shown", "candidate", "both"],
        default="none",
        help="If --captions-dir is set, include caption text for: "
        "'shown' frames in the current round, 'candidate' unseen frame IDs, or 'both'.",
    )
    parser.add_argument(
        "--caption-max-chars",
        type=int,
        default=200,
        help="Max characters per caption snippet included in the prompt (0 disables truncation).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Defaults to http://host:port.")
    parser.add_argument("--model-id", default=None, help="Explicit remote model ID for chat completions.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--start-server", action="store_true", help="Start vLLM server subprocess")
    parser.add_argument(
        "--restart-server-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --start-server, restart vLLM if a request fails (e.g., timeout).",
    )
    parser.add_argument(
        "--server-log",
        default=None,
        help="Optional file path to append vLLM server logs (stdout+stderr).",
    )
    parser.add_argument("--server-timeout-s", type=int, default=600)
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument(
        "--max-retries-per-round",
        type=int,
        default=0,
        help="Retry budget when the model output is missing required tags or requests invalid frames (default: 0 = no retries).",
    )
    parser.add_argument(
        "--strict-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate the sample immediately on illegal actions (e.g., invalid <select>, missing required tags, <think>); format issues like invalid <summarize> do not terminate.",
    )
    parser.add_argument(
        "--fallback-on-invalid-candidate-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using candidate frame IDs, fall back to heuristic sampling if the model outputs out-of-range IDs. "
        "Disable to make out-of-range candidate IDs count as an illegal action (and terminate when --strict-actions is enabled).",
    )
    parser.add_argument("--log-jsonl", default=os.getenv("REVISE_LOG_PATH", "debug_prompt_logs/revise_samples.jsonl"))
    parser.add_argument("--summary-json", default=None, help="Optional path to save a run summary JSON.")
    parser.add_argument("--progress-interval", type=int, default=10)
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
        help="If no <answer> is produced within max rounds, issue a final answer-only request (matches ReviseAgentLoop).",
    )
    parser.add_argument(
        "--answer-only-final-round",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, reject <answer> outputs before the final round; the model must request frames until max_rounds.",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Log eval metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "verl-revise"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", ""))
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", ""))
    args = parser.parse_args()

    # Candidate-ID mode requires candidate frames to exist.
    if bool(getattr(args, "use_candidate_frame_ids", False)):
        args.use_candidate_frames = True
    if bool(getattr(args, "require_candidate_frames", False)):
        args.use_candidate_frames = True

    if getattr(args, "caption_include", "none") != "none":
        if not getattr(args, "captions_dir", None):
            raise ValueError("--caption-include requires --captions-dir to be set.")
        if not os.path.isdir(str(args.captions_dir)):
            raise ValueError(f"--captions-dir does not exist or is not a directory: {args.captions_dir}")

    if getattr(args, "observation_mode", "image") == "caption":
        if not getattr(args, "captions_dir", None):
            raise ValueError("--observation-mode caption requires --captions-dir to be set.")
        if getattr(args, "caption_include", "none") == "none":
            # Caption-only REVISE needs at least shown captions to be meaningful.
            args.caption_include = "shown"
    if args.base_url and args.start_server:
        raise ValueError("--base-url cannot be combined with --start-server.")

    random.seed(args.seed)
    rng = random.Random(args.seed)

    server_proc: Optional[subprocess.Popen[str]] = None
    try:
        if args.start_server:
            server_proc = _start_vllm_server(args)
            wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
            wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)

        base_url = resolve_base_url(args.base_url, args.host, args.port)
        model_id = get_model_id(base_url, model_id=args.model_id)

        samples = _load_nextqa_samples(
            csv_path=args.csv,
            map_json=args.map_json,
            video_root=args.video_root,
            max_samples=args.max_samples,
            seed=args.seed or 42,
        )
        if not samples:
            raise RuntimeError("No samples loaded (check csv/map/video_root).")

        num_shards = max(1, int(getattr(args, "num_shards", 1)))
        shard_idx = int(getattr(args, "shard_idx", 0))
        if num_shards > 1:
            if not (0 <= shard_idx < num_shards):
                raise ValueError(f"--shard-idx must be in [0, {num_shards}) (got {shard_idx}).")

            def _suffix_path(path: str, *, suffix: str) -> str:
                root, ext = os.path.splitext(path)
                if ext:
                    return f"{root}{suffix}{ext}"
                return f"{path}{suffix}"

            # Avoid multiple shards writing to the same output files by default.
            shard_suffix = f".shard{shard_idx}of{num_shards}"
            if args.log_jsonl and shard_suffix not in args.log_jsonl:
                args.log_jsonl = _suffix_path(args.log_jsonl, suffix=shard_suffix)
            if args.summary_json and shard_suffix not in args.summary_json:
                args.summary_json = _suffix_path(args.summary_json, suffix=shard_suffix)

            samples = [s for i, s in enumerate(samples) if (i % num_shards) == shard_idx]
            if not samples:
                raise RuntimeError(
                    f"No samples selected for shard {shard_idx}/{num_shards} (check --max-samples / sharding)."
                )

        resume_completed = 0
        correct = 0
        total_rounds = 0
        if args.resume_from_log and args.log_jsonl and os.path.exists(args.log_jsonl):
            resume_completed, correct, total_rounds = _load_progress_from_log(
                args.log_jsonl, max_rounds=args.max_rounds
            )
            resume_completed = min(resume_completed, len(samples))

        processed = resume_completed
        failed = 0
        run = maybe_init_wandb(
            args,
            run_config={
                "task": "revise_plug_and_play_nextqa_vllm",
                "dataset_csv": args.csv,
                "video_root": args.video_root,
                "map_json": args.map_json,
                "model_path": args.model_path,
                "engine": "vllm",
                "observation_mode": getattr(args, "observation_mode", "image"),
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": args.dtype,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_samples": args.max_samples,
                "num_shards": int(getattr(args, "num_shards", 1)),
                "shard_idx": int(getattr(args, "shard_idx", 0)),
                "max_rounds": args.max_rounds,
                "max_frames_per_round": args.max_frames_per_round,
                "max_retries_per_round": args.max_retries_per_round,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "use_candidate_frames": bool(args.use_candidate_frames),
                "use_candidate_frame_ids": bool(args.use_candidate_frame_ids),
                "require_candidate_frames": bool(getattr(args, "require_candidate_frames", False)),
                "captions_dir": getattr(args, "captions_dir", None),
                "caption_include": getattr(args, "caption_include", "none"),
                "caption_max_chars": int(getattr(args, "caption_max_chars", 0)),
                "strict_actions": bool(args.strict_actions),
                "force_final_answer": args.force_final_answer,
                "answer_only_final_round": bool(args.answer_only_final_round),
                "resume_from_log": args.resume_from_log,
                "initial_completed": resume_completed,
                "log_jsonl": args.log_jsonl,
            },
        )
        start_eval = time.time()
        invalid_outputs = 0
        invalid_action_terminated = 0
        total_model_calls = 0
        total_retries = 0
        fallback_frames_used = 0
        effective_rounds_total = 0
        total_frames_used = 0
        for sample in samples[resume_completed:]:
            processed += 1
            seen_frames: list[int] = []
            try:
                frame_count = sample.frame_count
                if frame_count <= 0 and getattr(args, "observation_mode", "image") != "caption":
                    try:
                        import decord

                        vr = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))
                        frame_count = int(len(vr))
                    except Exception:
                        frame_count = 0

                question_block = _format_question(sample.question, sample.choices)
                prompt_template = (
                    DEFAULT_SYSTEM_PROMPT_CAPTION_ONLY
                    if getattr(args, "observation_mode", "image") == "caption"
                    else DEFAULT_SYSTEM_PROMPT
                )
                system_prompt = prompt_template.format(max_frames_per_round=args.max_frames_per_round)

                video_captions: dict[int, str] = {}
                if getattr(args, "captions_dir", None) and getattr(args, "caption_include", "none") != "none":
                    video_captions = _load_video_captions(str(args.captions_dir), sample.video_id)

                summary_state = (
                    "P: I will summarize what has been shown so far; "
                    "O: I will record the key observations from the current evidence; "
                    "H: I will update my belief as new evidence arrives; "
                    "U: some key detail may still be unclear; "
                    "R: request more evidence if needed"
                )
                effective_rounds = 0
                terminated_reason: Optional[str] = None
                terminated_invalid_action = False

                # Caption-only mode uses caption indices (1fps) as the action space length L.
                observation_mode = getattr(args, "observation_mode", "image")
                fps = 0.0
                if observation_mode == "caption":
                    if video_captions:
                        frame_count = max(video_captions.keys(), default=-1) + 1
                    if frame_count <= 0:
                        # Fall back to a rough seconds estimate from video length, if available.
                        try:
                            import decord

                            vr = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))
                            video_len = int(len(vr))
                            fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
                            if fps and fps > 0 and video_len > 0:
                                frame_count = max(1, int(video_len / fps))
                        except Exception:
                            frame_count = max(1, int(sample.frame_count) if int(sample.frame_count) > 0 else 1)
                elif video_captions:
                    fps = _get_video_fps(sample.video_path)

                def _caption_for_index(idx: int) -> str:
                    if not video_captions:
                        return "[no caption]"
                    key = int(idx)
                    if observation_mode != "caption":
                        key = _caption_key_for_frame_index(int(idx), fps)
                    return video_captions.get(int(key)) or "[no caption]"

                init_frames = sample_uniform_indices(frame_count, args.max_frames_per_round)
                next_frames = [int(i) for i in init_frames if i >= 0]
                answer_letter: Optional[str] = None
                last_user_text: Optional[str] = None
                last_images: list[Image.Image] = []
                last_frames: list[int] = []

                for round_idx in range(1, args.max_rounds + 1):
                    # Frames shown in this round.
                    frames_this_round = [i for i in next_frames if i not in seen_frames]
                    if not frames_this_round:
                        frames_this_round = sample_uniform_indices(frame_count, 1)
                    frames_this_round = frames_this_round[: args.max_frames_per_round]
                    for i in frames_this_round:
                        if i not in seen_frames:
                            seen_frames.append(i)

                    candidate_next_frames: list[int] = []
                    if getattr(args, "use_candidate_frames", False):
                        k = args.candidate_k if args.candidate_k is not None else max(12, args.max_frames_per_round * 4)
                        candidate_next_frames = propose_candidate_frames(
                            frame_count=frame_count,
                            seen=set(seen_frames),
                            k=k,
                            rng=rng,
                        )
                    shown_captions: Optional[list[str]] = None
                    candidate_captions: Optional[list[str]] = None
                    shown_ts: Optional[list[int]] = None
                    candidate_ts: Optional[list[int]] = None
                    if video_captions:
                        include = getattr(args, "caption_include", "none")
                        max_chars = int(getattr(args, "caption_max_chars", 0))
                        if include in ("shown", "both"):
                            shown_captions = [
                                truncate_text(_caption_for_index(int(i)), max_chars)
                                for i in frames_this_round
                            ]
                            if observation_mode != "caption":
                                shown_ts = [_caption_key_for_frame_index(int(i), fps) for i in frames_this_round]
                        if include in ("candidate", "both") and candidate_next_frames:
                            candidate_captions = [
                                truncate_text(_caption_for_index(int(i)), max_chars)
                                for i in candidate_next_frames
                            ]
                            if observation_mode != "caption":
                                candidate_ts = [_caption_key_for_frame_index(int(i), fps) for i in candidate_next_frames]
                    images: list[Image.Image] = []
                    if observation_mode != "caption":
                        images = extract_frames(sample.video_path, frames_this_round)
                    user_text = _build_user_text(
                        question_block=question_block,
                        summary=summary_state,
                        frame_count=frame_count,
                        round_idx=round_idx,
                        frame_indices=frames_this_round,
                        seen_frames=seen_frames,
                        render_images=(observation_mode != "caption"),
                        hide_seen_frames=bool(getattr(args, "hide_seen_frames_in_prompt", False)),
                        candidate_unseen_frames=candidate_next_frames if getattr(args, "use_candidate_frames", False) else None,
                        use_candidate_frame_ids=bool(args.use_candidate_frame_ids),
                        require_candidate_frames=bool(getattr(args, "require_candidate_frames", False)),
                        shown_frame_captions=shown_captions,
                        candidate_id_captions=candidate_captions,
                        shown_frame_ts=shown_ts,
                        candidate_id_ts=candidate_ts,
                    )
                    if args.force_final_answer and round_idx >= args.max_rounds:
                        user_text = (
                            f"{user_text}\n\n"
                            "This is the final round. You MUST answer now using <think>...</think> then <answer>LETTER</answer>."
                        )
                    last_user_text = user_text
                    last_images = images
                    last_frames = frames_this_round

                    raw = ""
                    retry_feedback: Optional[str] = None
                    attempt_user_text = user_text
                    for retry_idx in range(max(0, int(args.max_retries_per_round)) + 1):
                        raw = _chat_once(
                            base_url=base_url,
                            model_id=model_id,
                            system_prompt=system_prompt,
                            user_text=attempt_user_text,
                            images=images,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens,
                            timeout_s=args.request_timeout_s,
                        )
                        total_model_calls += 1

                        frames_tag = extract_tag(raw, SELECT_RE)
                        requested_raw_frames: Optional[list[int]] = None
                        requested_mapped_frames: Optional[list[int]] = None
                        if frames_tag is not None:
                            requested_raw_frames = dedupe_preserve_order(parse_int_list(frames_tag))
                            if bool(args.use_candidate_frame_ids) and candidate_next_frames:
                                mapped: list[int] = []
                                invalid_id = False
                                for cid in requested_raw_frames:
                                    if 1 <= cid <= len(candidate_next_frames):
                                        mapped.append(int(candidate_next_frames[cid - 1]))
                                    else:
                                        invalid_id = True
                                requested_mapped_frames = None if invalid_id else dedupe_preserve_order(mapped)
                            else:
                                requested_mapped_frames = requested_raw_frames

                        maybe_log_jsonl(
                            args.log_jsonl,
                            {
                                "ts": time.time(),
                                "sample_id": sample.sample_id,
                                "qid": sample.qid,
                                "video_id": sample.video_id,
                                "video_path": sample.video_path,
                                "round_idx": round_idx,
                                "retry_idx": retry_idx,
                                "retry_feedback": retry_feedback,
                                "question": sample.question,
                                "choices": sample.choices,
                                "ground_truth_idx": sample.answer_idx,
                                "observation_mode": observation_mode,
                                "use_candidate_frames": bool(getattr(args, "use_candidate_frames", False)),
                                "use_candidate_frame_ids": bool(args.use_candidate_frame_ids),
                                "candidate_unseen_frames": candidate_next_frames if getattr(args, "use_candidate_frames", False) else None,
                                "captions_dir": getattr(args, "captions_dir", None),
                                "caption_include": getattr(args, "caption_include", "none"),
                                "caption_max_chars": int(getattr(args, "caption_max_chars", 0)),
                                "shown_frame_captions": shown_captions,
                                "candidate_id_captions": candidate_captions,
                                "seen_frames": seen_frames,
                                "current_frames": frames_this_round,
                                "requested_raw_frames": requested_raw_frames,
                                "requested_mapped_frames": requested_mapped_frames,
                                "summary_in": summary_state,
                                "system_prompt": system_prompt,
                                "user_text": attempt_user_text,
                                "raw_output": raw,
                            },
                        )

                        summary = extract_tag(raw, SUMMARIZE_RE)
                        if (
                            summary
                            and (not is_placeholder(summary))
                            and (not contains_banned_example(summary))
                            and summary_has_ohrpu(summary)
                            and (not _summary_has_stale_boilerplate(summary, seen_count=len(seen_frames)))
                        ):
                            summary_state = summary

                        think = extract_tag(raw, THINK_RE)
                        if think is None:
                            invalid_outputs += 1
                            terminated_reason = "missing_think"
                            if retry_idx < int(args.max_retries_per_round):
                                total_retries += 1
                                retry_feedback = _retry_feedback_text(
                                    "Invalid response: every response MUST begin with a <think>...</think> reasoning trace, "
                                    "then either <summarize> + <select> (request) or <answer> (final).",
                                    force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds),
                                )
                                attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                continue
                            if args.strict_actions:
                                invalid_action_terminated += 1
                                terminated_invalid_action = True
                                answer_letter = None
                                break
                            # Fall back to the usual next_frames heuristic.
                            fallback_frames_used += 1
                            requested = _sample_unseen_frames(
                                frame_count, set(seen_frames), args.max_frames_per_round, rng=rng
                            )
                            next_frames = (
                                requested[: args.max_frames_per_round]
                                if requested
                                else sample_uniform_indices(frame_count, 1)
                            )
                            break

                        answer = extract_tag(raw, ANSWER_RE)
                        if answer:
                            answer_letter = normalize_answer_letter(answer, len(sample.choices))
                            if answer_letter is None:
                                invalid_outputs += 1
                                terminated_reason = "invalid_answer_letter"
                                if retry_idx < int(args.max_retries_per_round):
                                    total_retries += 1
                                    retry_feedback = _retry_feedback_text(
                                        "Invalid response: <answer> must be exactly ONE option letter (A/B/C/D/E). "
                                        "Do not output words or a sentence.",
                                        force_answer=True,
                                    )
                                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                    continue
                                if args.strict_actions:
                                    invalid_action_terminated += 1
                                    terminated_invalid_action = True
                                    answer_letter = None
                                    break
                                # Non-strict: ignore the invalid answer and continue with a fallback frame.
                                fallback_frames_used += 1
                                next_frames = _sample_unseen_frames(
                                    frame_count, set(seen_frames), args.max_frames_per_round, rng=rng
                                )
                                if not next_frames:
                                    next_frames = sample_uniform_indices(frame_count, 1)
                                answer_letter = None
                                break

                            if bool(args.answer_only_final_round) and round_idx < args.max_rounds:
                                invalid_outputs += 1
                                terminated_reason = "early_answer_disallowed"
                                if retry_idx < int(args.max_retries_per_round):
                                    total_retries += 1
                                    retry_feedback = _retry_feedback_text(
                                        "Invalid response: do NOT answer yet. Request more frames using <summarize>...</summarize> and <select>...</select>.",
                                        force_answer=False,
                                    )
                                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                    answer_letter = None
                                    continue
                                if args.strict_actions:
                                    invalid_action_terminated += 1
                                    terminated_invalid_action = True
                                    answer_letter = None
                                    break
                                # Non-strict: ignore the early answer and continue with a fallback frame request.
                                fallback_frames_used += 1
                                next_frames = _sample_unseen_frames(
                                    frame_count, set(seen_frames), args.max_frames_per_round, rng=rng
                                )
                                if not next_frames:
                                    next_frames = sample_uniform_indices(frame_count, 1)
                                answer_letter = None
                                break

                            # Strict-paper Answer round: <think> + <answer> only. No <summarize>
                            # is required here; the last committed summary is reused as the state
                            # (captured above when a valid <summarize> was present on a Select round).
                            break

                        frames_text = extract_tag(raw, SELECT_RE)

                        # If we didn't answer, we must request frames, with a valid summary.
                        if frames_text is None:
                            invalid_outputs += 1
                            terminated_reason = "missing_frames_tag"
                            if retry_idx < int(args.max_retries_per_round):
                                total_retries += 1
                                retry_feedback = _retry_feedback_text(
                                    "Invalid response: missing <select> tag for requesting more frames. "
                                    "Remember: <select> must list NEW frame indices to view NEXT (not already seen).",
                                    force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds),
                                )
                                attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                continue
                            if args.strict_actions:
                                invalid_action_terminated += 1
                                terminated_invalid_action = True
                                answer_letter = None
                                break
                            next_frames = sample_uniform_indices(frame_count, 1)
                            break

                        if summary is None or is_placeholder(summary) or contains_banned_example(summary) or (not summary_has_ohrpu(summary)):
                            invalid_outputs += 1
                            terminated_reason = "invalid_select_summary"
                            if retry_idx < int(args.max_retries_per_round):
                                total_retries += 1
                                retry_feedback = _retry_feedback_text(
                                    "Invalid response: include a meaningful <summarize> with P/O/H/U/R in that exact order "
                                    "(no placeholders like '.../none/unknown').",
                                    force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds),
                                )
                                attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                continue
                        if summary is not None and _summary_has_stale_boilerplate(summary, seen_count=len(seen_frames)):
                            invalid_outputs += 1
                            terminated_reason = "stale_select_summary"
                            if retry_idx < int(args.max_retries_per_round):
                                total_retries += 1
                                retry_feedback = _retry_feedback_text(
                                    "Invalid response: the <summarize> claims no frames/captions were seen, but evidence was shown. "
                                    "Rewrite <summarize> to reflect what was observed so far (P/O/H/U/R), then request frames.",
                                    force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds),
                                )
                                attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                continue

                        if (not bool(args.use_candidate_frame_ids)) and _frames_has_range_syntax(frames_text):
                            invalid_outputs += 1
                            terminated_reason = "frames_range_syntax"
                            if retry_idx < int(args.max_retries_per_round):
                                total_retries += 1
                                retry_feedback = _retry_feedback_text(
                                    "Invalid response: <select> must be a comma-separated list of integers only "
                                    "(NO ranges like '4-182', no hyphens). Choose up to {k} NEW frames.".format(
                                        k=args.max_frames_per_round
                                    ),
                                    force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds),
                                )
                                attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                continue

                        requested = dedupe_preserve_order(parse_int_list(frames_text))
                        if bool(args.use_candidate_frame_ids) and candidate_next_frames:
                            mapped: list[int] = []
                            invalid_id = False
                            for cid in requested:
                                if 1 <= cid <= len(candidate_next_frames):
                                    mapped.append(int(candidate_next_frames[cid - 1]))
                                else:
                                    invalid_id = True
                            if invalid_id:
                                invalid_outputs += 1
                                terminated_reason = "frames_out_of_range"
                                if retry_idx < int(args.max_retries_per_round):
                                    total_retries += 1
                                    retry_feedback = _retry_feedback_text(
                                        "Invalid response: when Candidate Frame IDs are provided, <select> must contain only "
                                        "IDs in the allowed range [1..K] (comma-separated integers).",
                                        force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds),
                                    )
                                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                    continue
                                if args.strict_actions and not bool(args.fallback_on_invalid_candidate_ids):
                                    invalid_action_terminated += 1
                                    terminated_invalid_action = True
                                    answer_letter = None
                                    break
                                # Be forgiving: fall back to heuristic sampling instead of hard-terminating.
                                fallback_frames_used += 1
                                requested = candidate_next_frames[: args.max_frames_per_round]
                                if not requested:
                                    requested = _sample_unseen_frames(
                                        frame_count, set(seen_frames), args.max_frames_per_round, rng=rng
                                    )
                                next_frames = (
                                    requested[: args.max_frames_per_round]
                                    if requested
                                    else sample_uniform_indices(frame_count, 1)
                                )
                                break
                            else:
                                requested = dedupe_preserve_order(mapped)
                                requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames]
                        else:
                            if bool(getattr(args, "require_candidate_frames", False)) and candidate_next_frames:
                                allowed = {int(i) for i in candidate_next_frames}
                                disallowed = [i for i in requested if int(i) not in allowed]
                                if disallowed:
                                    invalid_outputs += 1
                                    terminated_reason = "frames_not_in_candidates"
                                    if retry_idx < int(args.max_retries_per_round):
                                        total_retries += 1
                                        retry_feedback = _retry_feedback_text(
                                            "Invalid response: requested frames must be chosen ONLY from the candidate unseen frame list/ranges provided.",
                                            force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds),
                                        )
                                        attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                        continue
                                    if args.strict_actions:
                                        invalid_action_terminated += 1
                                        terminated_invalid_action = True
                                        answer_letter = None
                                        break
                                    requested = []
                                else:
                                    requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames and int(i) in allowed]
                            else:
                                allowed_ranges = unseen_intervals(frame_count, seen_frames)
                                requested = [
                                    i
                                    for i in requested
                                    if 0 <= i < frame_count and i not in seen_frames and in_intervals(i, allowed_ranges)
                                ]

                        if requested and len(requested) > int(args.max_frames_per_round):
                            invalid_outputs += 1
                            terminated_reason = "too_many_frames"
                            requested = requested[: int(args.max_frames_per_round)]
                        if not requested:
                            invalid_outputs += 1
                            terminated_reason = "invalid_frames"
                            if retry_idx < int(args.max_retries_per_round):
                                total_retries += 1
                                candidate_text = (
                                    " Allowed unseen ranges: "
                                    f"{format_intervals(unseen_intervals(frame_count, seen_frames))}."
                                )
                                retry_feedback = _retry_feedback_text(
                                    "Invalid response: requested frames must be NEW and within range. "
                                    "In <select>, output 1–{k} comma-separated integers NOT in Seen frames.".format(
                                        k=args.max_frames_per_round
                                    )
                                    + candidate_text,
                                    force_answer=bool(args.force_final_answer and round_idx >= args.max_rounds),
                                )
                                attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                                continue
                            if args.strict_actions:
                                invalid_action_terminated += 1
                                terminated_invalid_action = True
                                answer_letter = None
                                break

                            # Fall back to heuristic sampling.
                            fallback_frames_used += 1
                            requested = candidate_next_frames[: args.max_frames_per_round]
                            if not requested:
                                requested = _sample_unseen_frames(
                                    frame_count, set(seen_frames), args.max_frames_per_round, rng=rng
                                )
                            next_frames = (
                                requested[: args.max_frames_per_round] if requested else sample_uniform_indices(frame_count, 1)
                            )
                            break

                        next_frames = requested[: args.max_frames_per_round]
                        effective_rounds += 1
                        effective_rounds_total += 1
                        break

                    if answer_letter is not None:
                        break
                    if args.strict_actions and terminated_invalid_action:
                        break

                total_rounds += round_idx
                if (
                    args.force_final_answer
                    and answer_letter is None
                    and last_user_text is not None
                    and not (args.strict_actions and terminated_invalid_action)
                ):
                    forced_user_text = (
                        f"{last_user_text}\n\n"
                        "Max rounds reached. Provide the final answer now using <think>...</think> then <answer>LETTER</answer>."
                    )
                    raw = _chat_once(
                        base_url=base_url,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_text=forced_user_text,
                        images=last_images,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.request_timeout_s,
                    )
                    total_model_calls += 1
                    maybe_log_jsonl(
                        args.log_jsonl,
                        {
                            "ts": time.time(),
                            "sample_id": sample.sample_id,
                            "qid": sample.qid,
                            "video_id": sample.video_id,
                            "video_path": sample.video_path,
                            "round_idx": args.max_rounds + 1,
                            "forced_answer": True,
                            "question": sample.question,
                            "choices": sample.choices,
                        "ground_truth_idx": sample.answer_idx,
                        "seen_frames": seen_frames,
                        "current_frames": last_frames,
                        "summary_in": summary_state,
                        "system_prompt": system_prompt,
                        "user_text": forced_user_text,
                        "raw_output": raw,
                    },
                )
                answer = extract_tag(raw, ANSWER_RE)
                if answer:
                    answer_letter = normalize_answer_letter(answer, len(sample.choices))

                if answer_letter is not None:
                    pred_idx = ord(answer_letter) - ord("A")
                    if pred_idx == sample.answer_idx:
                        correct += 1
                total_frames_used += len(seen_frames)
            except Exception as e:
                failed += 1
                total_rounds += args.max_rounds
                if (
                    args.start_server
                    and args.restart_server_on_failure
                    and server_proc is not None
                    and isinstance(e, requests.exceptions.RequestException)
                ):
                    try:
                        stop_server(server_proc)
                    except Exception:
                        pass
                    server_proc = _start_vllm_server(args)
                    wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
                    wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
                    model_id = get_model_id(base_url, model_id=args.model_id)

            if args.progress_interval > 0 and processed % args.progress_interval == 0:
                elapsed = time.time() - start_eval
                acc = correct / max(1, processed)
                avg_rounds = total_rounds / max(1, processed)
                calls_per_sample = total_model_calls / max(1, processed)
                avg_frames_used = total_frames_used / max(1, processed)
                print(
                    f"[{processed}/{len(samples)}] acc={acc:.4f} avg_rounds={avg_rounds:.3f} "
                    f"failed={failed} invalid={invalid_outputs} retries={total_retries} "
                    f"calls={total_model_calls} calls/sample={calls_per_sample:.2f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
                wandb_log(
                    run,
                    {
                        "eval/acc": acc,
                        "eval/avg_rounds": avg_rounds,
                        "eval/avg_effective_rounds": effective_rounds_total / max(1, processed),
                        "eval/avg_frames_used": avg_frames_used,
                        "eval/failed": failed,
                        "eval/processed": processed,
                        "eval/elapsed_s": elapsed,
                        "eval/invalid_outputs": invalid_outputs,
                        "eval/invalid_action_terminated": invalid_action_terminated,
                        "eval/total_retries": total_retries,
                        "eval/total_model_calls": total_model_calls,
                        "eval/fallback_frames_used": fallback_frames_used,
                        "eval/calls_per_sample": calls_per_sample,
                    },
                    step=processed,
                )

        acc = correct / max(1, processed)
        avg_rounds = total_rounds / max(1, processed)
        elapsed = time.time() - start_eval
        prompt_log_lines = _count_file_lines(args.log_jsonl) if args.log_jsonl else 0
        prompt_log_bytes = os.path.getsize(args.log_jsonl) if args.log_jsonl and os.path.exists(args.log_jsonl) else 0
        avg_frames_used = total_frames_used / max(1, processed)
        results = {
            "samples": processed,
            "correct": correct,
            "accuracy": acc,
            "total_rounds": total_rounds,
            "avg_rounds": avg_rounds,
            "total_frames_used": total_frames_used,
            "avg_frames_used": avg_frames_used,
            "total_effective_rounds": effective_rounds_total,
            "avg_effective_rounds": effective_rounds_total / max(1, processed),
            "failed": failed,
            "elapsed_s": elapsed,
            "prompt_log_lines": prompt_log_lines,
            "prompt_log_bytes": prompt_log_bytes,
            "invalid_outputs": invalid_outputs,
            "invalid_action_terminated": invalid_action_terminated,
            "total_retries": total_retries,
            "total_model_calls": total_model_calls,
            "fallback_frames_used": fallback_frames_used,
        }
        print(json.dumps(results, indent=2))
        wandb_log(
            run,
            {
                "eval/final_acc": acc,
                "eval/final_avg_rounds": avg_rounds,
                "eval/final_avg_effective_rounds": effective_rounds_total / max(1, processed),
                "eval/final_avg_frames_used": avg_frames_used,
                "eval/final_failed": failed,
                "eval/final_elapsed_s": elapsed,
                "eval/prompt_log_lines": prompt_log_lines,
                "eval/prompt_log_bytes": prompt_log_bytes,
                "eval/final_invalid_outputs": invalid_outputs,
                "eval/final_invalid_action_terminated": invalid_action_terminated,
                "eval/final_total_retries": total_retries,
                "eval/final_total_model_calls": total_model_calls,
                "eval/final_fallback_frames_used": fallback_frames_used,
            },
            step=processed,
        )
        if run is not None:
            run.summary["prompt_log_jsonl"] = args.log_jsonl
            run.summary["prompt_log_lines"] = prompt_log_lines
            run.summary["prompt_log_bytes"] = prompt_log_bytes
            run.summary["final_acc"] = acc
            run.summary["final_avg_rounds"] = avg_rounds
            run.summary["final_avg_effective_rounds"] = effective_rounds_total / max(1, processed)
            run.summary["final_avg_frames_used"] = avg_frames_used
            run.summary["final_failed"] = failed
            run.summary["invalid_outputs"] = invalid_outputs
            run.summary["invalid_action_terminated"] = invalid_action_terminated
            run.summary["total_retries"] = total_retries
            run.summary["total_model_calls"] = total_model_calls
            run.summary["fallback_frames_used"] = fallback_frames_used
            run.finish()

        if args.summary_json:
            summary_dir = os.path.dirname(args.summary_json)
            if summary_dir:
                os.makedirs(summary_dir, exist_ok=True)
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": "revise_plug_and_play_nextqa_vllm",
                        "dataset_csv": args.csv,
                        "video_root": args.video_root,
                        "map_json": args.map_json,
                        "model_path": args.model_path,
                        "engine": "vllm",
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "dtype": args.dtype,
                        "max_model_len": args.max_model_len,
                        "gpu_memory_utilization": args.gpu_memory_utilization,
                        "max_samples": args.max_samples,
                        "seed": int(getattr(args, "seed", 0)),
                        "num_shards": int(getattr(args, "num_shards", 1)),
                        "shard_idx": int(getattr(args, "shard_idx", 0)),
                        "max_rounds": args.max_rounds,
                        "max_frames_per_round": args.max_frames_per_round,
                        "max_retries_per_round": args.max_retries_per_round,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "use_candidate_frames": bool(args.use_candidate_frames),
                        "use_candidate_frame_ids": bool(args.use_candidate_frame_ids),
                        "require_candidate_frames": bool(getattr(args, "require_candidate_frames", False)),
                        "strict_actions": bool(args.strict_actions),
                        "fallback_on_invalid_candidate_ids": bool(args.fallback_on_invalid_candidate_ids),
                        "answer_only_final_round": bool(args.answer_only_final_round),
                        "results": results,
                        "prompt_log_jsonl": args.log_jsonl,
                        "prompt_log_lines": prompt_log_lines,
                        "prompt_log_bytes": prompt_log_bytes,
                        "wandb": {
                            "enabled": bool(args.use_wandb),
                            "mode": args.wandb_mode
                            or os.getenv("WANDB_MODE")
                            or ("online" if os.getenv("WANDB_API_KEY") else "offline"),
                            "project": args.wandb_project,
                            "entity": args.wandb_entity,
                            "name": args.wandb_name,
                            "group": args.wandb_group,
                            "id": getattr(run, "id", None),
                            "run_dir": getattr(run, "dir", None),
                            "url": getattr(run, "url", None),
                        },
                        "command": "python " + " ".join(os.sys.argv),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        return 0
    finally:
        if server_proc is not None:
            stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
