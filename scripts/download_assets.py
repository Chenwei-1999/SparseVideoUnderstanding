#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

VIDEOESPRESSO_TEST_REPO = "hshjerry0315/VideoEspresso-Test"
VIDEOESPRESSO_TRAIN_REPO = "hshjerry0315/VideoEspresso_train_video"
VIDEOMME_REPO = "lmms-lab/Video-MME"
NEXTQA_HF_MIRROR_REPO = "VLM2Vec/nextqa"
NEXTQA_RAW_VIDEO_HF_REPO = "rhymes-ai/NeXTVideo"
NEXTQA_GITHUB_REPO = "https://github.com/doc-doc/NExT-QA.git"
QWEN25_VL_3B_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN25_VL_7B_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"


def _find_first(root: Path, names: tuple[str, ...]) -> str | None:
    if not root.exists():
        return None
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return str(matches[0])
    return None


def _snapshot_download(
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    *,
    dry_run: bool,
    allow_patterns: list[str] | None = None,
) -> str:
    if dry_run:
        return str(local_dir)
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,
    )


def _write_env(path: Path, exports: dict[str, str | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Source this file before running scripts/doctor.py or paper_suite.py."]
    for key, value in exports.items():
        if value:
            lines.append(f"export {key}={json.dumps(str(value))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_gdown(url: str, output: Path, *, dry_run: bool) -> dict[str, Any]:
    gdown = shutil.which("gdown")
    match = re.search(r"/file/d/([^/]+)/", url)
    resolved_url = f"https://drive.google.com/uc?id={match.group(1)}&export=download" if match else url
    info = {
        "url": url,
        "resolved_url": resolved_url,
        "output": str(output),
        "tool": gdown,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if dry_run or not gdown:
        return info
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if not os.access(Path(env.get("HOME", "")).expanduser(), os.W_OK):
        cache_home = output.parent / ".gdown_home"
        cache_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(cache_home)
    proc = subprocess.run(
        [gdown, "--continue", resolved_url, "-O", str(output)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    info.update({"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})
    return info


def _clone_nextqa_annotations(nextqa_root: Path, *, dry_run: bool) -> dict[str, Any]:
    repo_dir = nextqa_root / "_official_repo"
    out = {"repo": NEXTQA_GITHUB_REPO, "repo_dir": str(repo_dir), "returncode": None, "stdout": "", "stderr": ""}
    if dry_run:
        return out
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", NEXTQA_GITHUB_REPO, str(repo_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        out.update({"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})
        if proc.returncode != 0:
            return out
    src = repo_dir / "dataset" / "nextqa"
    dst = nextqa_root / "nextqa"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "val.csv", "test.csv", "map_vid_vidorID.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    if (dst / "map_vid_vidorID.json").exists():
        shutil.copy2(dst / "map_vid_vidorID.json", nextqa_root / "map_vid_vidorID.json")
    out["annotations_dir"] = str(dst)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or register external assets for REVISE reproduction.")
    parser.add_argument(
        "--asset-root",
        default=os.getenv("REVISE_ASSET_ROOT", str(REPO_ROOT / "data" / "revise_assets")),
        help="Ignored-by-git directory used for datasets, model snapshots, and generated env file.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--videoespresso", action="store_true")
    parser.add_argument("--videomme", action="store_true")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--nextqa-links", action="store_true", help="Record official NExT-QA Google Drive links.")
    parser.add_argument(
        "--nextqa-hf-mirror",
        action="store_true",
        help="Clone official NExT-QA annotations and download the VLM2Vec/nextqa eval-video mirror.",
    )
    parser.add_argument(
        "--nextqa-raw-hf",
        action="store_true",
        help="Clone official NExT-QA annotations and download the rhymes-ai/NeXTVideo raw-video archive.",
    )
    parser.add_argument("--all", action="store_true", help="Download/register every supported asset group.")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(asset_root / ".hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(asset_root / ".hf_home" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(asset_root / ".hf_home" / "datasets"))
    os.environ.setdefault("HF_XET_CACHE", str(asset_root / ".hf_home" / "xet"))
    selected_all = bool(args.all)
    report: dict[str, Any] = {
        "asset_root": str(asset_root),
        "dry_run": bool(args.dry_run),
        "downloads": {},
        "manual": {},
        "exports": {},
    }
    exports: dict[str, str | None] = {}

    if selected_all or args.videoespresso:
        ve_root = asset_root / "VideoEspresso"
        test_dir = ve_root / "test_video"
        train_dir = ve_root / "train_video"
        report["downloads"]["videoespresso_test"] = _snapshot_download(
            VIDEOESPRESSO_TEST_REPO, "dataset", test_dir, dry_run=bool(args.dry_run)
        )
        report["downloads"]["videoespresso_train_video"] = _snapshot_download(
            VIDEOESPRESSO_TRAIN_REPO, "dataset", train_dir, dry_run=bool(args.dry_run)
        )
        exports.update(
            {
                "REVISE_VIDEOESPRESSO_ROOT": str(ve_root),
                "REVISE_VIDEOESPRESSO_TEST_VIDEO_ROOT": str(test_dir),
                "REVISE_VIDEOESPRESSO_TEST_JSON": _find_first(test_dir, ("bench_hard.json", "bench_final.json")),
                "REVISE_VIDEOESPRESSO_TRAIN_VIDEO_JSON": _find_first(
                    train_dir,
                    ("videoespresso_train_video.json", "videoespresso_train.json"),
                ),
            }
        )

    if selected_all or args.videomme:
        videomme_dir = asset_root / "Video-MME"
        report["downloads"]["videomme"] = _snapshot_download(
            VIDEOMME_REPO,
            "dataset",
            videomme_dir,
            dry_run=bool(args.dry_run),
            allow_patterns=["README.md", ".gitattributes", "videomme/*", "subtitle.zip"],
        )
        exports["REVISE_VIDEO_CACHE_DIR"] = str(asset_root / "video_cache")

    if selected_all or args.models:
        model_root = asset_root / "models"
        qwen3b = model_root / "Qwen2.5-VL-3B-Instruct"
        qwen7b = model_root / "Qwen2.5-VL-7B-Instruct"
        report["downloads"]["qwen25_vl_3b"] = _snapshot_download(
            QWEN25_VL_3B_REPO, "model", qwen3b, dry_run=bool(args.dry_run)
        )
        report["downloads"]["qwen25_vl_7b"] = _snapshot_download(
            QWEN25_VL_7B_REPO, "model", qwen7b, dry_run=bool(args.dry_run)
        )
        exports.update(
            {
                "REVISE_QWEN25_VL_3B_PATH": str(qwen3b),
                "REVISE_QWEN25_VL_7B_PATH": str(qwen7b),
            }
        )

    if args.nextqa_hf_mirror or args.nextqa_raw_hf:
        nextqa_root = asset_root / "NExT-QA"
        report["downloads"]["nextqa_annotations"] = _clone_nextqa_annotations(nextqa_root, dry_run=bool(args.dry_run))
        if args.nextqa_hf_mirror:
            videos_zip = nextqa_root / "videos.zip"
            if args.dry_run:
                report["downloads"]["nextqa_hf_videos_zip"] = str(videos_zip)
            else:
                from huggingface_hub import hf_hub_download

                nextqa_root.mkdir(parents=True, exist_ok=True)
                report["downloads"]["nextqa_hf_videos_zip"] = hf_hub_download(
                    NEXTQA_HF_MIRROR_REPO,
                    "videos.zip",
                    repo_type="dataset",
                    local_dir=str(nextqa_root),
                )
        if args.nextqa_raw_hf:
            raw_zip = nextqa_root / "NExTVideo.zip"
            if args.dry_run:
                report["downloads"]["nextqa_raw_hf_videos_zip"] = str(raw_zip)
            else:
                from huggingface_hub import hf_hub_download

                nextqa_root.mkdir(parents=True, exist_ok=True)
                report["downloads"]["nextqa_raw_hf_videos_zip"] = hf_hub_download(
                    NEXTQA_RAW_VIDEO_HF_REPO,
                    "NExTVideo.zip",
                    repo_type="dataset",
                    local_dir=str(nextqa_root),
                    resume_download=True,
                )
        exports.update(
            {
                "REVISE_NEXTQA_ROOT": str(nextqa_root),
                "REVISE_NEXTQA_VIDEO_ROOT": str(nextqa_root / "NExTVideo"),
                "REVISE_NEXTQA_MAP_JSON": str(nextqa_root / "map_vid_vidorID.json"),
                "REVISE_NEXTQA_TRAIN_CSV": str(nextqa_root / "nextqa" / "train.csv"),
                "REVISE_NEXTQA_VAL_CSV": str(nextqa_root / "nextqa" / "val.csv"),
            }
        )

    if selected_all or args.nextqa_links:
        nextqa_root = asset_root / "NExT-QA"
        report["manual"]["nextqa"] = {
            "repo": "https://github.com/doc-doc/NExT-QA.git",
            "raw_videos": _maybe_gdown(
                "https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view",
                nextqa_root / "NExTVideo.zip",
                dry_run=True,
            ),
            "test_data": _maybe_gdown(
                "https://drive.google.com/file/d/1_MEqDeQHc8Y8Uw7eW58HVuZy2iyThILQ/view?usp=sharing",
                nextqa_root / "nextqa_test_data.zip",
                dry_run=True,
            ),
            "note": "NExT-QA requires accepting/downloading Google Drive assets; clone the official repo for annotations, then set the exports below.",
        }
        exports.update(
            {
                "REVISE_NEXTQA_ROOT": str(nextqa_root),
                "REVISE_NEXTQA_VIDEO_ROOT": str(nextqa_root / "NExTVideo"),
                "REVISE_NEXTQA_MAP_JSON": str(nextqa_root / "map_vid_vidorID.json"),
                "REVISE_NEXTQA_TRAIN_CSV": str(nextqa_root / "nextqa" / "train.csv"),
                "REVISE_NEXTQA_VAL_CSV": str(nextqa_root / "nextqa" / "val.csv"),
            }
        )

    report["exports"] = exports
    env_path = asset_root / "revise_env.sh"
    _write_env(env_path, exports)
    report["env_file"] = str(env_path)

    manifest_path = asset_root / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "env_file": str(env_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
