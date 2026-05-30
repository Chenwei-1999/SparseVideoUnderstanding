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

from revise.plug_and_play_egoschema_vllm import _load_egoschema_samples
from revise.plug_and_play_nextqa_vllm import (
    _chat_once,
    _load_nextqa_samples,
    _start_vllm_server,
)
from revise.pnp_utils import (
    extract_frames,
    extract_video_info,
    format_question_block,
    format_videoespresso_question_block,
    get_model_id,
    maybe_log_jsonl,
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

    correct = 0
    failed = 0
    total_calls = 0
    total_frames = 0
    start_t = time.time()
    system_prompt = ""

    for i, sample in enumerate(samples, start=1):
        frame_indices: list[int] = []
        raw_output = ""
        user_text = ""
        if dataset_name == "videoespresso" and args.videoespresso_use_official_prompt:
            question_block = format_videoespresso_question_block(
                sample.question,
                sample.choices,
                task=getattr(sample, "task", ""),
                evidence=getattr(sample, "evidence", ""),
                with_evidence=bool(args.videoespresso_with_evidence),
                revise_answer_tags=False,
            )
        else:
            question_block = format_question_block(sample.question, sample.choices)
        try:
            frame_count, _ = extract_video_info(sample.video_path)
            frame_indices = sample_uniform_indices(int(frame_count or 0), int(args.max_frames))
            images = extract_frames(sample.video_path, frame_indices)
            if not images:
                raise RuntimeError("no frames extracted")
            user_text = _build_user_text(question_block, frame_indices)
            raw_output = _chat_once(
                base_url=base_url,
                model_id=model_id,
                system_prompt=system_prompt,
                user_text=user_text,
                images=images,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens),
                timeout_s=int(args.timeout_s),
            )
            total_calls += 1
        except Exception as e:
            if args.restart_server_on_failure and args.start_server and server_proc is not None:
                try:
                    stop_server(server_proc)
                except Exception:
                    pass
                server_proc = _start_vllm_server(args)
                wait_port(args.host, args.port, timeout_s=args.server_timeout_s)
                wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
                model_id = get_model_id(base_url, model_id=args.model_id)
                try:
                    frame_count, _ = extract_video_info(sample.video_path)
                    frame_indices = sample_uniform_indices(int(frame_count or 0), int(args.max_frames))
                    images = extract_frames(sample.video_path, frame_indices)
                    if not images:
                        raise RuntimeError("no frames extracted")
                    user_text = _build_user_text(question_block, frame_indices)
                    raw_output = _chat_once(
                        base_url=base_url,
                        model_id=model_id,
                        system_prompt=system_prompt,
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
                maybe_log_jsonl(
                    args.log_jsonl,
                    {
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
                        ],
                        "frame_indices": frame_indices,
                        "videoespresso_official_prompt": bool(
                            dataset_name == "videoespresso" and args.videoespresso_use_official_prompt
                        ),
                        "videoespresso_with_evidence": bool(
                            dataset_name == "videoespresso" and args.videoespresso_with_evidence
                        ),
                        "error": f"{type(e).__name__}: {e}",
                    },
                )
                continue

        pred = normalize_answer_letter(raw_output, len(sample.choices))
        gt = normalize_answer_letter(chr(ord("A") + int(sample.answer_idx)), len(sample.choices))
        is_correct = pred is not None and gt is not None and pred == gt
        if is_correct:
            correct += 1
        total_frames += len(frame_indices)

        maybe_log_jsonl(
            args.log_jsonl,
            {
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
                "videoespresso_official_prompt": bool(
                    dataset_name == "videoespresso" and args.videoespresso_use_official_prompt
                ),
                "videoespresso_with_evidence": bool(
                    dataset_name == "videoespresso" and args.videoespresso_with_evidence
                ),
            },
        )

        if i % 50 == 0:
            answered = i - failed
            acc = correct / max(1, answered)
            print(
                f"[{i}/{len(samples)}] dataset={dataset_name} acc={acc:.4f} failed={failed} calls={total_calls}",
                flush=True,
            )

    if server_proc is not None:
        try:
            stop_server(server_proc)
        except Exception:
            pass

    answered = max(0, len(samples) - failed)
    summary = {
        "task": "oneshot_local_mc_vllm",
        "dataset": dataset_name,
        "samples": len(samples),
        "answered": answered,
        "correct": correct,
        "accuracy": float(correct / max(1, answered)),
        "failed": failed,
        "avg_frames": float(total_frames / max(1, answered)),
        "elapsed_s": float(time.time() - start_t),
        "total_model_calls": total_calls,
        "log_jsonl": args.log_jsonl,
        "videoespresso_use_official_prompt": bool(args.videoespresso_use_official_prompt)
        if dataset_name == "videoespresso"
        else None,
        "videoespresso_with_evidence": bool(args.videoespresso_with_evidence)
        if dataset_name == "videoespresso"
        else None,
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload, flush=True)
    if args.summary_json:
        out_dir = os.path.dirname(args.summary_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    if len(samples) > 0 and (failed >= len(samples) or total_calls == 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
