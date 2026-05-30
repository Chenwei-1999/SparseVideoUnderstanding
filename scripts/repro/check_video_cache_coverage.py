#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.revise.plug_and_play_videomme_lvbench_vllm import (  # noqa: E402
    _load_lvbench_samples,
    _load_videomme_samples,
)


def _dataset_split(dataset: str, split: str) -> str:
    return split or ("test" if dataset == "videomme" else "train")


def _coverage(dataset: str, split: str, cache_root: Path) -> dict[str, Any]:
    samples = _load_videomme_samples(split) if dataset == "videomme" else _load_lvbench_samples(split)
    video_keys = sorted({sample.video_key for sample in samples})
    cache_dir = cache_root / dataset
    missing: list[str] = []
    present = 0
    bytes_total = 0
    for key in video_keys:
        path = cache_dir / key
        if path.exists() and path.stat().st_size > 0:
            present += 1
            bytes_total += path.stat().st_size
        else:
            missing.append(key)
    return {
        "dataset": dataset,
        "split": split,
        "cache_dir": str(cache_dir),
        "total_videos": len(video_keys),
        "present": present,
        "missing": len(missing),
        "coverage": present / len(video_keys) if video_keys else 0.0,
        "bytes_total": bytes_total,
        "missing_examples": missing[:20],
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| dataset | split | present | total | missing | coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {split} | {present} | {total_videos} | {missing} | {coverage:.2%} |".format(**row)
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check cached video coverage for HF video benchmarks.")
    parser.add_argument("--dataset", choices=["videomme", "lvbench"], action="append")
    parser.add_argument("--split", default="", help="Defaults to Video-MME test or LVBench train per dataset.")
    parser.add_argument(
        "--video-cache-dir",
        default=os.getenv("REVISE_VIDEO_CACHE_DIR", str(REPO_ROOT / "data" / "revise_assets" / "video_cache")),
    )
    parser.add_argument("--json", default="", help="Optional JSON output path.")
    parser.add_argument("--md", default="", help="Optional Markdown output path.")
    args = parser.parse_args()

    asset_root = REPO_ROOT / "data" / "revise_assets"
    os.environ.setdefault("HF_HOME", str(asset_root / ".hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(asset_root / ".hf_home" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(asset_root / ".hf_home" / "datasets"))
    os.environ.setdefault("HF_XET_CACHE", str(asset_root / ".hf_home" / "xet"))

    cache_root = Path(args.video_cache_dir).expanduser().resolve()
    rows = [
        _coverage(dataset, _dataset_split(dataset, args.split), cache_root)
        for dataset in (args.dataset or ["videomme", "lvbench"])
    ]
    payload = {"video_cache_dir": str(cache_root), "rows": rows}
    print(_markdown(rows))
    if args.json:
        path = Path(args.json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.md:
        path = Path(args.md).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown(rows) + "\n", encoding="utf-8")
    return 0 if all(row["missing"] == 0 for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
