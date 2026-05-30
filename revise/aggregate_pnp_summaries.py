#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any


def _expand_paths(paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for p in paths:
        if any(ch in p for ch in ["*", "?", "["]):
            expanded.extend(sorted(glob.glob(p)))
            continue
        if os.path.isdir(p):
            expanded.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
            continue
        expanded.append(p)
    # De-dup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in expanded:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(p)
    return out


def _get_results(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary.get("results")
    if not isinstance(results, dict):
        raise ValueError("summary missing 'results' dict")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="Summary JSON paths, globs, or directories (reads *.json).",
    )
    parser.add_argument("--output", default=None, help="Optional path to write aggregated JSON.")
    args = parser.parse_args()

    paths = _expand_paths(args.paths)
    if not paths:
        raise SystemExit("No summary files found.")

    summaries: list[dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))

    totals = {
        "samples": 0,
        "correct": 0,
        "total_rounds": 0,
        "total_effective_rounds": 0,
        "failed": 0,
        "invalid_outputs": 0,
        "invalid_action_terminated": 0,
        "total_retries": 0,
        "total_model_calls": 0,
        "fallback_frames_used": 0,
        "prompt_log_lines": 0,
        "prompt_log_bytes": 0,
    }

    shard_details: list[dict[str, Any]] = []
    for path, summary in zip(paths, summaries, strict=False):
        results = _get_results(summary)
        shard_details.append(
            {
                "path": path,
                "dataset_csv": summary.get("dataset_csv"),
                "num_shards": summary.get("num_shards"),
                "shard_idx": summary.get("shard_idx"),
                "results": results,
                "wandb_url": (summary.get("wandb") or {}).get("url"),
                "prompt_log_jsonl": summary.get("prompt_log_jsonl"),
            }
        )
        totals["samples"] += int(results.get("samples") or 0)
        totals["correct"] += int(results.get("correct") or 0)
        totals["total_rounds"] += int(results.get("total_rounds") or 0)
        totals["total_effective_rounds"] += int(results.get("total_effective_rounds") or 0)
        totals["failed"] += int(results.get("failed") or 0)
        totals["invalid_outputs"] += int(results.get("invalid_outputs") or 0)
        totals["invalid_action_terminated"] += int(results.get("invalid_action_terminated") or 0)
        totals["total_retries"] += int(results.get("total_retries") or 0)
        totals["total_model_calls"] += int(results.get("total_model_calls") or 0)
        totals["fallback_frames_used"] += int(results.get("fallback_frames_used") or 0)
        totals["prompt_log_lines"] += int(summary.get("prompt_log_lines") or 0)
        totals["prompt_log_bytes"] += int(summary.get("prompt_log_bytes") or 0)

    denom = max(1, totals["samples"])
    aggregated = {
        "task": "revise_plug_and_play_nextqa_vllm_aggregate",
        "summaries": shard_details,
        "results": {
            **totals,
            "accuracy": float(totals["correct"] / denom),
            "avg_rounds": float(totals["total_rounds"] / denom),
            "avg_effective_rounds": float(totals["total_effective_rounds"] / denom),
            "calls_per_sample": float(totals["total_model_calls"] / denom),
        },
    }

    payload = json.dumps(aggregated, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()

