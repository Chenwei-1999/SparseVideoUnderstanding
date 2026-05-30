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
from revise.plug_and_play_videomme_lvbench_vllm import (
    LVBenchDataset,
    VideoMMEDataset,
    VllmHttpBackend,
    _chat_once,
    _load_lvbench_samples,
    _load_videomme_samples,
    _start_vllm_server,
)
from revise.pnp_protocols import LoopConfig, RunStats
from revise.pnp_utils import (
    extract_frames_1fps as _extract_frames_1fps,
    extract_video_info as _extract_video_info,
    format_question_block as _format_question_block,
    format_videomme_question_block as _format_videomme_question_block,
    get_model_id as _get_model_id,
    maybe_log_jsonl as _maybe_log_jsonl,
    normalize_answer_letter as _normalize_answer_letter,
    parse_time_reference_range as _parse_time_reference_range,
    pick_free_port as _pick_free_port,
    resolve_base_url as _resolve_base_url,
    sample_uniform_indices_inclusive as _sample_uniform_indices_inclusive,
    shard_by_video as _shard_by_video,
    stop_server as _stop_server,
    timeline_len_1fps as _timeline_len_1fps,
    wait_for_server as _wait_for_server,
)


def _build_user_text(question_block: str, num_frames: int) -> str:
    lines: list[str] = []
    lines.append(question_block)
    lines.append("")
    lines.append(f"You will be shown {num_frames} video frames sampled at 1 fps.")
    lines.append("Answer with EXACTLY ONE option letter (e.g., A/B/C/D). Do not output any other text.")
    lines.append("")
    lines.append("Frames:")
    for i in range(num_frames):
        lines.append(f"Frame {i+1}: <image>")
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
    ap.add_argument("--cached-only", action="store_true", help="Skip samples whose videos are not cached locally.")
    ap.add_argument(
        "--allow-missing-cached-videos",
        action="store_true",
        help="Allow full cached-only runs to evaluate a cached subset when videos are missing.",
    )
    ap.add_argument(
        "--videomme-use-official-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the official Video-MME no-subtitle multiple-choice prompt template.",
    )

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
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.7)

    ap.add_argument("--max-frames", type=int, default=5)
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
        args.port = _pick_free_port()

    # Reuse vLLM server launcher from plug_and_play_videomme_lvbench_vllm.py, which expects
    # `max_frames_per_round` for `--limit-mm-per-prompt`. Map our oneshot `--max-frames` to it.
    if not hasattr(args, "max_frames_per_round"):
        setattr(args, "max_frames_per_round", int(args.max_frames))

    split = args.split
    if not split:
        split = "test" if args.dataset == "videomme" else "train"

    if args.dataset == "videomme":
        samples = _load_videomme_samples(split)
    else:
        samples = _load_lvbench_samples(split)

    cache_dir = Path(args.video_cache_dir) / args.dataset
    cache_filter_total = len(samples)
    cache_filter_missing = 0
    cache_filter_missing_examples: list[str] = []
    if args.cached_only:
        filtered = []
        missing_keys: list[str] = []
        for s in samples:
            p = cache_dir / s.video_key
            try:
                ok = p.stat().st_size > 0
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

    samples = _shard_by_video(samples, args.num_shards, args.shard_idx)
    samples.sort(key=lambda s: (s.video_key, s.uid))
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

    resume_completed = 0
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
        resume_completed = len(seen)
        if resume_completed:
            print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}", flush=True)
            samples = samples[resume_completed:]

    base_url = _resolve_base_url(args.base_url, args.host, args.port)

    server_proc = None
    if args.start_server:
        server_proc = _start_vllm_server(args)
        _wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)

    model_id = _get_model_id(base_url, model_id=args.model_id)

    rng = random.Random(1337 + int(args.shard_idx))

    # Build the shared Dataset adapter + Backend and delegate the single-round
    # eval loop to the shared harness (setting="oneshot_baseline"). The adapter
    # reuses extract_frames / initial_frame_indices / normalize_answer / the
    # one-shot prompt; the harness owns video-existence skip, scoring, logging,
    # and the launcher's own top-level summary schema.
    dataset_adapter: Any
    if args.dataset == "videomme":
        dataset_adapter = VideoMMEDataset(
            split=split,
            video_cache_dir=args.video_cache_dir,
            yt_dlp_timeout_s=600,
            videomme_use_official_prompt=bool(args.videomme_use_official_prompt),
        )
    else:
        dataset_adapter = LVBenchDataset(
            split=split,
            video_cache_dir=args.video_cache_dir,
            yt_dlp_timeout_s=600,
            videomme_use_official_prompt=bool(args.videomme_use_official_prompt),
        )

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
        seed=1337 + int(args.shard_idx),
    )

    def _restart_server() -> Optional[str]:
        nonlocal server_proc, model_id
        if not (args.restart_server_on_failure and args.start_server and server_proc is not None):
            raise RuntimeError("restart_not_configured")
        try:
            _stop_server(server_proc)
        except Exception:
            pass
        server_proc = _start_vllm_server(args)
        _wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
        model_id = _get_model_id(base_url, model_id=args.model_id)
        return model_id

    restart_enabled = bool(args.restart_server_on_failure and args.start_server)
    backend = VllmHttpBackend(restart_server=None)

    args._pnp_setting = "oneshot_baseline"
    args._pnp_split = split
    args._pnp_oneshot_cache_dir = cache_dir
    args._pnp_oneshot_resume_completed = resume_completed
    args._pnp_oneshot_cache_filter_total = cache_filter_total
    args._pnp_oneshot_cache_filter_missing = cache_filter_missing
    args._pnp_oneshot_cache_filter_missing_examples = cache_filter_missing_examples
    args._pnp_oneshot_restart_server = _restart_server if restart_enabled else None

    stats = RunStats()
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

    if args.start_server and server_proc is not None:
        try:
            _stop_server(server_proc)
        except Exception:
            pass

    total = int(results.get("samples", 0))
    if total > 0 and (
        int(results.get("failed", 0)) >= total or int(results.get("total_model_calls", 0)) == 0
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
