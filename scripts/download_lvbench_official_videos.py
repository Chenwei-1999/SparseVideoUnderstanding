#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LVBENCH_REPO = "lmms-lab/LVBench"
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


def _parse_chunks(values: list[str]) -> list[int]:
    chunks: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                chunks.update(range(int(lo), int(hi) + 1))
            else:
                chunks.add(int(part))
    bad = [chunk for chunk in chunks if chunk < 1 or chunk > 14]
    if bad:
        raise SystemExit(f"LVBench chunks must be in 1..14, got {bad}")
    return sorted(chunks)


def _extract_videos(zip_path: Path, cache_dir: Path, *, dry_run: bool, overwrite_existing: bool) -> dict[str, Any]:
    if dry_run and not zip_path.exists():
        return {
            "zip": str(zip_path),
            "members": None,
            "extracted": None,
            "skipped_existing": None,
            "sample_members": [],
        }
    extracted = 0
    skipped = 0
    members: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in VIDEO_EXTS:
                continue
            members.append(info.filename)
            out_path = cache_dir / Path(info.filename).name
            if out_path.exists() and out_path.stat().st_size > 0 and not overwrite_existing:
                skipped += 1
                continue
            if dry_run:
                extracted += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with zf.open(info) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            tmp_path.replace(out_path)
            extracted += 1
    return {
        "zip": str(zip_path),
        "members": len(members),
        "extracted": extracted,
        "skipped_existing": skipped,
        "sample_members": members[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and flatten official LVBench HF video chunks.")
    parser.add_argument("--chunk", action="append", default=[], help="Chunk number/range, e.g. 1, 001, 1-4. Default: all.")
    parser.add_argument(
        "--asset-root",
        default=os.getenv("REVISE_ASSET_ROOT", str(REPO_ROOT / "data" / "revise_assets")),
    )
    parser.add_argument(
        "--video-cache-dir",
        default=os.getenv("REVISE_VIDEO_CACHE_DIR", str(REPO_ROOT / "data" / "revise_assets" / "video_cache")),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-zip-after-extract", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    raw_dir = asset_root / "LVBench"
    cache_dir = Path(args.video_cache_dir).expanduser().resolve() / "lvbench"
    chunks = _parse_chunks(args.chunk) if args.chunk else list(range(1, 15))

    os.environ.setdefault("HF_HOME", str(asset_root / ".hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(asset_root / ".hf_home" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(asset_root / ".hf_home" / "datasets"))
    os.environ.setdefault("HF_XET_CACHE", str(asset_root / ".hf_home" / "xet"))

    report: dict[str, Any] = {
        "repo": LVBENCH_REPO,
        "asset_root": str(asset_root),
        "raw_dir": str(raw_dir),
        "video_cache_dir": str(cache_dir),
        "chunks": [],
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        from huggingface_hub import hf_hub_download

    for chunk in chunks:
        filename = f"video_chunks/videos_chunk_{chunk:03d}.zip"
        zip_path = raw_dir / filename
        entry: dict[str, Any] = {"chunk": chunk, "filename": filename, "zip_path": str(zip_path)}
        if args.dry_run:
            entry["download"] = "dry_run"
        else:
            raw_dir.mkdir(parents=True, exist_ok=True)
            entry["download"] = hf_hub_download(
                LVBENCH_REPO,
                filename,
                repo_type="dataset",
                local_dir=str(raw_dir),
                resume_download=True,
            )
        if zip_path.exists() or args.dry_run:
            entry["extract"] = _extract_videos(
                zip_path,
                cache_dir,
                dry_run=bool(args.dry_run),
                overwrite_existing=bool(args.overwrite_existing),
            )
            if args.delete_zip_after_extract and not args.dry_run and entry["extract"]["members"] > 0:
                zip_path.unlink()
                entry["deleted_zip"] = True
        else:
            entry["error"] = "zip_not_found_after_download"
        report["chunks"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    manifest = Path(args.manifest).expanduser() if args.manifest else raw_dir / "lvbench_official_video_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [entry for entry in report["chunks"] if entry.get("error")]
    print(json.dumps({"manifest": str(manifest), "chunks": len(chunks), "failed": len(failed)}, ensure_ascii=False, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
