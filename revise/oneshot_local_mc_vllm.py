#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Allow direct execution via `python examples/...py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise import pnp_harness
from revise.plug_and_play_egoschema_vllm import _load_egoschema_samples
from revise.plug_and_play_nextqa_vllm import (
    _chat_once,
    _load_nextqa_samples,
    _start_vllm_server,
)
from revise.pnp_protocols import LoopConfig, RunStats
from revise.pnp_utils import (
    extract_frames,
    extract_video_info,
    format_question_block,
    format_videoespresso_question_block,
    get_model_id,
    normalize_answer_letter,
    pick_free_port,
    resolve_base_url,
    sample_uniform_indices,
    shard_by_video,
    stop_server,
    wait_port,
    wait_for_server,
)


def _build_user_text(question_block: str, frame_indices: list[int]) -> str:
    lines = [question_block, ""]
    lines.append(f"You will be shown {len(frame_indices)} video frames.")
    lines.append("Answer with EXACTLY ONE option letter (for example: A/B/C/D/E). Do not output any other text.")
    lines.append("")
    lines.append("Frames:")
    for idx in frame_indices:
        lines.append(f"Frame {idx}: <image>")
    return "\n".join(lines)


class LocalMCDataset:
    """Single-round (one-shot) adapter for local multiple-choice datasets.

    Wraps the NExT-QA / jsonmc (EgoSchema-style) loaders and exposes only the
    methods exercised by :func:`revise.pnp_engine.run_sample_oneshot`. Frame
    probing/extraction and chat go through this module's bare-name helpers so
    the patch surface matches the legacy standalone loop.

    The one-shot prompt enumerates frames by their *actual* sampled timeline
    index (``Frame {idx}: <image>``), so :meth:`oneshot_user_text` consumes the
    ``frame_indices`` the engine threads through.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        videoespresso_use_official_prompt: bool,
        videoespresso_with_evidence: bool,
    ) -> None:
        self.dataset_name = str(dataset_name).strip().lower()
        self.videoespresso_use_official_prompt = bool(videoespresso_use_official_prompt)
        self.videoespresso_with_evidence = bool(videoespresso_with_evidence)

    def video_path(self, sample: Any) -> str:
        return sample.video_path

    def frame_count(self, sample: Any) -> int:
        # Legacy parity: probe the video for its frame count; the engine then
        # uniformly samples within it. A 0-frame probe is treated as an empty
        # timeline by the engine (counted as failed).
        frame_count, _ = extract_video_info(sample.video_path)
        return int(frame_count or 0)

    def num_choices(self, sample: Any) -> int:
        return len(sample.choices)

    def normalize_answer(self, sample: Any, answer_text: str) -> Optional[str]:
        return normalize_answer_letter(answer_text, self.num_choices(sample))

    def ground_truth_letter(self, sample: Any) -> Optional[str]:
        return normalize_answer_letter(chr(ord("A") + int(sample.answer_idx)), self.num_choices(sample))

    def is_correct(self, sample: Any, pred_letter: str) -> bool:
        gt = self.ground_truth_letter(sample)
        return bool(pred_letter and gt and pred_letter == gt)

    def format_question(self, sample: Any) -> str:
        if self.dataset_name == "videoespresso" and self.videoespresso_use_official_prompt:
            return format_videoespresso_question_block(
                sample.question,
                sample.choices,
                task=getattr(sample, "task", ""),
                evidence=getattr(sample, "evidence", ""),
                with_evidence=bool(self.videoespresso_with_evidence),
                revise_answer_tags=False,
            )
        return format_question_block(sample.question, sample.choices)

    def initial_frame_indices(self, sample: Any, frame_count: int, cfg: LoopConfig) -> list[int]:
        _ = sample
        return sample_uniform_indices(int(frame_count or 0), int(cfg.max_frames_per_round))

    def extract_frames(self, sample: Any, indices: list[int]) -> list[Any]:
        return extract_frames(sample.video_path, indices)

    def oneshot_user_text(
        self,
        question_block: str,
        num_frames: int,
        *,
        frame_indices: Optional[list[int]] = None,
    ) -> str:
        _ = num_frames
        return _build_user_text(question_block, list(frame_indices or []))


class _LocalMCBackend:
    """vLLM HTTP backend that routes through this module's bare-name ``_chat_once``.

    Mirrors the legacy standalone call so the same chat seam is patched in tests.
    """

    def chat(self, **kwargs: Any) -> str:
        return _chat_once(**kwargs)

    def get_model_id(self, base_url: str, model_id: Optional[str] = None) -> str:
        return get_model_id(base_url, model_id=model_id)


def _slice_samples(samples: list[Any], start_idx: int, end_idx: int, max_samples: int) -> list[Any]:
    start_idx = max(0, int(start_idx or 0))
    end_idx = int(end_idx or 0)
    if end_idx <= 0:
        end_idx = len(samples)
    if start_idx > 0 or end_idx < len(samples):
        samples = samples[start_idx:end_idx]
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nextqa", "jsonmc"], required=True)
    ap.add_argument("--dataset-name", default="", help="Dataset label for logging when --dataset jsonmc is used.")

    ap.add_argument("--video-root", required=True)
    ap.add_argument("--map-json", default=None, help="Required for --dataset nextqa.")
    ap.add_argument("--csv", default=None, help="Required for --dataset nextqa.")
    ap.add_argument("--json", default=None, help="Required for --dataset jsonmc.")

    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

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
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)

    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument(
        "--videoespresso-use-official-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For dataset-name=videoespresso, format task/options prompts like the official close-ended evaluator.",
    )
    ap.add_argument(
        "--videoespresso-with-evidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For dataset-name=videoespresso, include the row evidence field in the prompt.",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout-s", type=int, default=120)

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)

    ap.add_argument("--log-jsonl", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--resume-from-log", action="store_true")
    args = ap.parse_args()

    if args.base_url and args.start_server:
        raise ValueError("--base-url cannot be combined with --start-server.")
    if args.port <= 0:
        args.port = pick_free_port()
    if not hasattr(args, "max_frames_per_round"):
        setattr(args, "max_frames_per_round", int(args.max_frames))

    random.seed(args.seed)

    if args.dataset == "nextqa":
        if not args.csv or not args.map_json:
            raise ValueError("--dataset nextqa requires --csv and --map-json.")
        samples = _load_nextqa_samples(
            csv_path=args.csv,
            map_json=args.map_json,
            video_root=args.video_root,
            max_samples=0,
            seed=args.seed or 42,
        )
        dataset_name = "nextqa"
    else:
        if not args.json:
            raise ValueError("--dataset jsonmc requires --json.")
        samples = _load_egoschema_samples(args.json, args.video_root, max_samples=0, seed=args.seed or 42)
        dataset_name = str(args.dataset_name).strip().lower() or "jsonmc"

    samples = _slice_samples(samples, args.start_idx, args.end_idx, args.max_samples)
    samples = shard_by_video(samples, args.num_shards, args.shard_idx, video_key_attr="video_path")
    if not samples:
        raise SystemExit("No samples selected.")

    if args.num_shards > 1:
        suffix = f".shard{args.shard_idx}of{args.num_shards}"

        def _suffix_path(path: str) -> str:
            root, ext = os.path.splitext(path)
            return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"

        if args.log_jsonl and suffix not in args.log_jsonl:
            args.log_jsonl = _suffix_path(args.log_jsonl)
        if args.summary_json and suffix not in args.summary_json:
            args.summary_json = _suffix_path(args.summary_json)

    if args.resume_from_log and args.log_jsonl and os.path.exists(args.log_jsonl):
        seen = set()
        with open(args.log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("sample_id")
                if sid and obj.get("pred_answer"):
                    seen.add(str(sid))
        if seen:
            samples = [s for s in samples if str(getattr(s, "sample_id", "")) not in seen]

    server_proc = None
    if args.start_server:
        server_proc = _start_vllm_server(args)
        wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
        wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)

    base_url = resolve_base_url(args.base_url, args.host, args.port)
    model_id = get_model_id(base_url, model_id=args.model_id)

    num_samples = len(samples)
    system_prompt = ""

    # Build the shared single-round (one-shot) Dataset adapter + the vLLM HTTP
    # backend, then delegate the eval loop to the shared harness
    # (setting="oneshot_baseline"). The adapter reuses this module's bare-name
    # frame/chat helpers; the harness owns scoring, logging, and this launcher's
    # own summary schema.
    #
    # Behavioral deltas vs. the legacy standalone loop (consistent with the
    # other one-shot migrations): video probing/extraction failures are counted
    # as failed by the engine (instead of being inline-skipped) and the engine
    # owns the single chat per sample. Missing videos are NOT pre-skipped here
    # (skip_missing=False) so extraction drives the failure exactly as before.
    dataset_adapter = LocalMCDataset(
        dataset_name=dataset_name,
        videoespresso_use_official_prompt=bool(args.videoespresso_use_official_prompt),
        videoespresso_with_evidence=bool(args.videoespresso_with_evidence),
    )
    backend = _LocalMCBackend()
    cfg = LoopConfig(
        max_rounds=1,
        max_frames_per_round=int(args.max_frames),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        request_timeout_s=int(args.timeout_s),
        max_retries_per_round=0,
        strict_actions=False,
        force_final_answer=False,
        use_candidate_frames=False,
        candidate_k=None,
        use_candidate_frame_ids=False,
        require_candidate_frames=False,
        answer_only_final_round=False,
        observation_mode="image",
        caption_include="none",
        caption_max_chars=0,
        captions_dir=None,
        hide_seen_frames_in_prompt=False,
        log_jsonl=args.log_jsonl or None,
        seed=args.seed,
    )

    is_videoespresso_official = bool(
        dataset_name == "videoespresso" and args.videoespresso_use_official_prompt
    )
    is_videoespresso_evidence = bool(
        dataset_name == "videoespresso" and args.videoespresso_with_evidence
    )

    def _restart_server() -> Optional[str]:
        nonlocal server_proc, model_id
        if not (args.restart_server_on_failure and args.start_server and server_proc is not None):
            raise RuntimeError("restart_not_configured")
        try:
            stop_server(server_proc)
        except Exception:
            pass
        server_proc = _start_vllm_server(args)
        wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
        wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
        model_id = get_model_id(base_url, model_id=args.model_id)
        return model_id

    restart_enabled = bool(args.restart_server_on_failure and args.start_server)

    def _log_record(
        sample: Any,
        *,
        outcome: Any,
        pred: Optional[str],
        is_correct: bool,
        video_path: str,
        split: str,
    ) -> dict[str, Any]:
        _ = split
        frame_indices = list(outcome.frame_indices)
        question_block = dataset_adapter.format_question(sample)
        user_text = _build_user_text(question_block, frame_indices)
        gt = normalize_answer_letter(chr(ord("A") + int(sample.answer_idx)), len(sample.choices))
        raw_output = outcome.raw_output
        return {
            "ts": time.time(),
            "dataset": dataset_name,
            "sample_id": sample.sample_id,
            "qid": getattr(sample, "qid", ""),
            "video_path": sample.video_path,
            "question": sample.question,
            "options": sample.choices,
            "task": getattr(sample, "task", ""),
            "evidence": getattr(sample, "evidence", ""),
            "system_prompt": system_prompt,
            "user_text": user_text,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": raw_output},
            ],
            "frame_indices": frame_indices,
            "pred_answer": pred,
            "answer_gt": gt,
            "correct": bool(is_correct),
            "raw_output": raw_output,
            "videoespresso_official_prompt": is_videoespresso_official,
            "videoespresso_with_evidence": is_videoespresso_evidence,
        }

    def _build_summary(
        *,
        samples_total: int,
        answered: int,
        correct: int,
        failed: int,
        invalid: int,
        frames_used: int,
        elapsed_s: float,
        stats: RunStats,
    ) -> dict[str, Any]:
        _ = invalid
        return {
            "task": "oneshot_local_mc_vllm",
            "dataset": dataset_name,
            "samples": samples_total,
            "answered": answered,
            "correct": correct,
            "accuracy": float(correct / max(1, answered)),
            "failed": failed,
            "avg_frames": float(frames_used / max(1, answered)),
            "elapsed_s": float(elapsed_s),
            "total_model_calls": stats.total_model_calls,
            "log_jsonl": args.log_jsonl,
            "videoespresso_use_official_prompt": bool(args.videoespresso_use_official_prompt)
            if dataset_name == "videoespresso"
            else None,
            "videoespresso_with_evidence": bool(args.videoespresso_with_evidence)
            if dataset_name == "videoespresso"
            else None,
        }

    args._pnp_setting = "oneshot_baseline"
    args._pnp_split = ""
    args._pnp_oneshot_resume_completed = 0
    args._pnp_oneshot_skip_missing_video = False
    args._pnp_oneshot_video_path = lambda sample: sample.video_path
    args._pnp_oneshot_log_record = _log_record
    args._pnp_oneshot_build_summary = _build_summary
    args._pnp_oneshot_restart_server = _restart_server if restart_enabled else None

    stats = RunStats()
    rng = random.Random(args.seed)
    results = pnp_harness.run_eval(
        samples,
        dataset=dataset_adapter,
        backend=backend,
        cfg=cfg,
        stats=stats,
        rng=rng,
        base_url=base_url,
        model_id=model_id,
        run=None,
        args=args,
    )

    if server_proc is not None:
        try:
            stop_server(server_proc)
        except Exception:
            pass

    failed = int(results.get("failed", 0))
    total_calls = int(results.get("total_model_calls", 0))
    if num_samples > 0 and (failed >= num_samples or total_calls == 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
