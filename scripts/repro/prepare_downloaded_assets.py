#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


def _run(cmd: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _extract_zip(zip_path: Path, out_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"zip": str(zip_path), "out_dir": str(out_dir), "dry_run": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("unzip"):
        # -DD skips restoring archived timestamps so extracted files get
        # current mtime — Quest scratch auto-purges by mtime and many
        # source zips carry vintage timestamps.
        res = _run(["unzip", "-q", "-n", "-DD", str(zip_path), "-d", str(out_dir)])
        _run(["find", str(out_dir), "-type", "f", "-exec", "touch", "{}", "+"])
        return res
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    # Python's zipfile.extractall also restores archived timestamps; refresh.
    _run(["find", str(out_dir), "-type", "f", "-exec", "touch", "{}", "+"])
    return {"zip": str(zip_path), "out_dir": str(out_dir), "python_zipfile": True, "returncode": 0}


def _count_media_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES)


def _prepare_videoespresso(asset_root: Path, *, dry_run: bool, include_train: bool) -> dict[str, Any]:
    ve_root = asset_root / "VideoEspresso"
    test_dir = ve_root / "test_video"
    train_dir = ve_root / "train_video"
    report: dict[str, Any] = {"root": str(ve_root), "steps": []}

    test_zip = test_dir / "all_video.zip"
    test_all_video = test_dir / "all_video"
    if test_zip.exists() and not any(test_all_video.glob("*")):
        report["steps"].append({"name": "extract_test_all_video", **_extract_zip(test_zip, test_dir, dry_run=dry_run)})

    split_zip = train_dir / "VideoEspresso_train_video.zip"
    split_parts = sorted(
        path
        for path in train_dir.glob("VideoEspresso_train_video.z*")
        if path.name.rsplit(".z", 1)[-1].isdigit()
    )
    merged_zip = train_dir / "VideoEspresso_train_video_merged.zip"
    train_all_video = train_dir / "all_video"
    train_media_count = _count_media_files(train_all_video)
    split_total_bytes = (
        (split_zip.stat().st_size if split_zip.exists() else 0)
        + sum(path.stat().st_size for path in split_parts if path.exists())
    )
    free_bytes = shutil.disk_usage(train_dir).free if train_dir.exists() else 0
    report["train_media_count"] = train_media_count
    report["train_split_archive_bytes"] = split_total_bytes
    report["train_free_bytes"] = free_bytes
    train_full_enough = train_media_count >= 20000

    if include_train and split_zip.exists() and split_parts and not train_full_enough and not merged_zip.exists():
        # Info-ZIP needs a merged archive before extraction, so full train extraction needs room
        # for both the merged zip and extracted media. Partial smoke extraction is handled by
        # extract_videoespresso_split_zip_subset.py and must not be mistaken for a full train set.
        minimum_bytes = split_total_bytes * 2
        if free_bytes < minimum_bytes and not dry_run:
            report["steps"].append(
                {
                    "name": "merge_train_split_zip",
                    "status": "blocked_insufficient_space",
                    "zip": str(split_zip),
                    "parts": len(split_parts),
                    "out": str(merged_zip),
                    "available_bytes": free_bytes,
                    "minimum_recommended_bytes": minimum_bytes,
                    "recommendation": "free space or use extract_videoespresso_split_zip_subset.py for smoke runs",
                }
            )
            report["paths"] = {
                "test_json": str(test_dir / "bench_hard.json"),
                "test_video_root": str(test_dir),
                "test_all_video": str(test_all_video),
                "train_json": str(train_dir / "videoespresso_train_video.json"),
                "train_video_root": str(train_dir),
                "train_all_video": str(train_all_video),
            }
            return report
        if dry_run:
            report["steps"].append(
                {
                    "name": "merge_train_split_zip",
                    "zip": str(split_zip),
                    "parts": len(split_parts),
                    "out": str(merged_zip),
                    "dry_run": True,
                }
            )
        else:
            report["steps"].append(
                {
                    "name": "merge_train_split_zip",
                    **_run(["zip", "-s", "0", str(split_zip), "--out", str(merged_zip)], cwd=train_dir),
                }
            )
    if include_train and merged_zip.exists() and not train_full_enough:
        report["steps"].append(
            {"name": "extract_train_all_video", **_extract_zip(merged_zip, train_dir, dry_run=dry_run)}
        )

    report["paths"] = {
        "test_json": str(test_dir / "bench_hard.json"),
        "test_video_root": str(test_dir),
        "test_all_video": str(test_all_video),
        "train_json": str(train_dir / "videoespresso_train_video.json"),
        "train_video_root": str(train_dir),
        "train_all_video": str(train_all_video),
    }
    return report


def _prepare_nextqa(asset_root: Path, *, dry_run: bool) -> dict[str, Any]:
    nextqa_root = asset_root / "NExT-QA"
    raw_zip = nextqa_root / "NExTVideo.zip"
    videos_zip = nextqa_root / "videos.zip"
    video_root = nextqa_root / "NExTVideo"
    report: dict[str, Any] = {"root": str(nextqa_root), "steps": []}
    if raw_zip.exists():
        report["steps"].append(
            {"name": "extract_nextqa_raw_videos", **_extract_zip(raw_zip, nextqa_root, dry_run=dry_run)}
        )
    elif videos_zip.exists() and not any(video_root.glob("*")):
        report["steps"].append(
            {"name": "extract_nextqa_eval_mirror_videos", **_extract_zip(videos_zip, video_root, dry_run=dry_run)}
        )
    report["paths"] = {
        "video_root": str(video_root),
        "map_json": str(nextqa_root / "map_vid_vidorID.json"),
        "train_csv": str(nextqa_root / "nextqa" / "train.csv"),
        "val_csv": str(nextqa_root / "nextqa" / "val.csv"),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract downloaded dataset archives into runnable layouts.")
    parser.add_argument(
        "--asset-root",
        default=os.getenv("REVISE_ASSET_ROOT", str(REPO_ROOT / "data" / "revise_assets")),
    )
    parser.add_argument("--videoespresso", action="store_true")
    parser.add_argument("--nextqa", action="store_true")
    parser.add_argument(
        "--videoespresso-train",
        action="store_true",
        help="Also merge/extract the large VideoEspresso train-video split archive.",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    report: dict[str, Any] = {"asset_root": str(asset_root), "dry_run": bool(args.dry_run)}
    if args.all or args.videoespresso:
        report["videoespresso"] = _prepare_videoespresso(
            asset_root,
            dry_run=bool(args.dry_run),
            include_train=bool(args.videoespresso_train),
        )
    if args.all or args.nextqa:
        report["nextqa"] = _prepare_nextqa(asset_root, dry_run=bool(args.dry_run))

    report_path = asset_root / "prepare_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
