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
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.revise.plug_and_play_videomme_lvbench_vllm import (
    _chat_once,
    _load_lvbench_samples,
    _load_videomme_samples,
    _start_vllm_server,
)
from examples.revise.pnp_utils import (
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
    start_t = time.time()

    correct = 0
    failed = 0
    total_calls = 0
    printed_error = False

    for i, sample in enumerate(samples, start=1):
        video_path = str(cache_dir / sample.video_key)
        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            failed += 1
            continue

        try:
            total_frames, fps = _extract_video_info(video_path)
            timeline_len = _timeline_len_1fps(total_frames, fps)
        except Exception:
            failed += 1
            continue

        if timeline_len <= 0:
            failed += 1
            continue

        range_start, range_end = 0, timeline_len - 1
        if args.dataset == "lvbench" and sample.time_reference:
            tr = _parse_time_reference_range(sample.time_reference, timeline_len)
            if tr is not None:
                range_start, range_end = tr

        frame_indices = _sample_uniform_indices_inclusive(range_start, range_end, int(args.max_frames))
        images = _extract_frames_1fps(video_path, frame_indices)
        if not images:
            failed += 1
            continue

        if args.dataset == "videomme" and args.videomme_use_official_prompt:
            question_block = _format_videomme_question_block(sample.question, sample.options)
        else:
            question_block = _format_question_block(sample.question, sample.options)
        user_text = _build_user_text(question_block, len(images))
        try:
            out = _chat_once(
                base_url=base_url,
                model_id=model_id,
                system_prompt="",
                user_text=user_text,
                images=images,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens),
                timeout_s=int(args.timeout_s),
            )
            total_calls += 1
        except Exception as e:
            if not printed_error:
                printed_error = True
                print(f"[first_error] {type(e).__name__}: {e}", flush=True)
            if args.restart_server_on_failure and args.start_server and server_proc is not None:
                try:
                    _stop_server(server_proc)
                except Exception:
                    pass
                server_proc = _start_vllm_server(args)
                _wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
                model_id = _get_model_id(base_url, model_id=args.model_id)
                try:
                    out = _chat_once(
                        base_url=base_url,
                        model_id=model_id,
                        system_prompt="",
                        user_text=user_text,
                        images=images,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_tokens=int(args.max_tokens),
                        timeout_s=int(args.timeout_s),
                    )
                    total_calls += 1
                except Exception:
                    failed += 1
                    continue
            else:
                failed += 1
                continue

        pred = _normalize_answer_letter(out, len(sample.options))
        gt = _normalize_answer_letter(sample.answer_letter, len(sample.options))
        is_correct = pred is not None and gt is not None and pred == gt
        if is_correct:
            correct += 1

        _maybe_log_jsonl(
            args.log_jsonl,
            {
                "ts": time.time(),
                "dataset": args.dataset,
                "split": split,
                "uid": sample.uid,
                "video_key": sample.video_key,
                "video_path": video_path,
                "time_reference": sample.time_reference,
                "sample_id": sample.sample_id,
                "question": sample.question,
                "options": sample.options,
                "answer_gt": sample.answer_letter,
                "pred_answer": pred,
                "raw_output": out,
                "correct": bool(is_correct),
            },
        )

        if i % 50 == 0:
            answered = i - failed
            acc = correct / max(1, answered)
            print(
                f"[{i}/{len(samples)}] acc={acc:.4f} failed={failed} calls={total_calls} elapsed_s={time.time()-start_t:.1f}",
                flush=True,
            )

    if args.start_server and server_proc is not None:
        try:
            _stop_server(server_proc)
        except Exception:
            pass

    total = len(samples) + resume_completed
    answered = max(0, total - failed)
    results = {
        "samples": total,
        "answered": answered,
        "correct": correct,
        "accuracy": float(correct / max(1, answered)),
        "failed": failed,
        "elapsed_s": float(time.time() - start_t),
        "total_model_calls": total_calls,
        "prompt_log_jsonl": args.log_jsonl,
        "cached_only": bool(args.cached_only),
        "allow_missing_cached_videos": bool(args.allow_missing_cached_videos),
        "cache_filter_total_samples": cache_filter_total,
        "cache_filter_missing_videos": cache_filter_missing,
        "cache_filter_missing_examples": cache_filter_missing_examples,
    }
    payload = json.dumps(results, indent=2, ensure_ascii=False)
    print(payload, flush=True)
    if args.summary_json:
        out_dir = os.path.dirname(args.summary_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    if total > 0 and (failed >= total or total_calls == 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
