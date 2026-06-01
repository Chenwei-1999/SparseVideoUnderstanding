#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise.datasets.lvbench import load_samples as load_lvbench_samples  # noqa: E402
from revise.datasets.video_download import download_youtube  # noqa: E402
from revise.datasets.videomme import load_samples as load_videomme_samples  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Populate the local video cache for Video-MME or LVBench.")
    ap.add_argument("--dataset", choices=["videomme", "lvbench"], required=True)
    ap.add_argument("--split", default="", help="Defaults to Video-MME test or LVBench train.")
    ap.add_argument(
        "--video-cache-dir",
        default=str(REPO_ROOT / "data" / "revise_assets" / "video_cache"),
        help="Cache root. Videos are saved under <root>/<dataset>/...",
    )
    ap.add_argument("--max-videos", type=int, default=0, help="Maximum distinct videos to cache. 0 means all.")
    ap.add_argument("--start-idx", type=int, default=0, help="Start offset after sorting distinct videos.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="", help="Where to write a JSON cache report.")
    ap.add_argument("--yt-dlp-timeout-s", type=int, default=900)
    args = ap.parse_args()

    asset_root = REPO_ROOT / "data" / "revise_assets"
    os.environ.setdefault("HF_HOME", str(asset_root / ".hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(asset_root / ".hf_home" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(asset_root / ".hf_home" / "datasets"))
    os.environ.setdefault("HF_XET_CACHE", str(asset_root / ".hf_home" / "xet"))

    split = args.split or ("test" if args.dataset == "videomme" else "train")
    samples = load_videomme_samples(split) if args.dataset == "videomme" else load_lvbench_samples(split)

    by_key: dict[str, Any] = {}
    for sample in samples:
        by_key.setdefault(sample.video_key, sample)
    video_items = [by_key[key] for key in sorted(by_key)]
    if args.start_idx > 0:
        video_items = video_items[int(args.start_idx) :]
    if args.max_videos > 0:
        video_items = video_items[: int(args.max_videos)]

    cache_dir = Path(args.video_cache_dir).expanduser().resolve() / args.dataset
    results: list[dict[str, Any]] = []
    started = time.time()
    for idx, sample in enumerate(video_items, start=1):
        out_path = cache_dir / sample.video_key
        record = {
            "idx": idx,
            "video_key": sample.video_key,
            "video_url": sample.video_url,
            "path": str(out_path),
            "status": "pending",
        }
        if out_path.exists() and out_path.stat().st_size > 0:
            record["status"] = "exists"
            record["bytes"] = out_path.stat().st_size
            results.append(record)
            continue
        if args.dry_run:
            record["status"] = "dry_run"
            results.append(record)
            continue
        try:
            download_youtube(
                sample.video_url,
                str(out_path),
                py_bin=sys.executable,
                timeout_s=int(args.yt_dlp_timeout_s),
            )
            record["status"] = "downloaded"
            record["bytes"] = out_path.stat().st_size if out_path.exists() else 0
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report = {
        "dataset": args.dataset,
        "split": split,
        "video_cache_dir": str(Path(args.video_cache_dir).expanduser().resolve()),
        "selected_videos": len(video_items),
        "downloaded": sum(1 for r in results if r["status"] == "downloaded"),
        "exists": sum(1 for r in results if r["status"] == "exists"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "dry_run": bool(args.dry_run),
        "elapsed_s": time.time() - started,
        "results": results,
    }
    manifest = (
        Path(args.manifest).expanduser()
        if args.manifest
        else cache_dir / f"{args.dataset}_{split}_cache_manifest.json"
    )
    _write_json(manifest, report)
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                **{k: report[k] for k in ("selected_videos", "downloaded", "exists", "failed", "dry_run")},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
