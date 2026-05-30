#!/usr/bin/env python3
"""Bootstrap a REVISE asset root from empty to fully populated, idempotently.

Designed for scratch volatility: re-runnable after a partial wipe, symlinks
pre-existing official-layout trees in EXISTING_* paths when present, and
falls back to downloads when not.

Components covered:
  - HF model snapshots (Qwen2.5-VL-{3B,7B,72B}, Qwen2-VL-7B, InternVL2-8B,
    LLaVA-OneVision-Qwen2-7B-OV)
  - VideoEspresso test_video (+ all_video extraction)
  - VideoEspresso train_video (multi-part split-zip; selective extraction
    is handled per-experiment elsewhere)
  - NExT-QA: NExTVideo.zip download + extract, annotations (train/val/test
    CSV + map_vid_vidorID.json) via doc-doc/NExT-QA mirror
  - EgoSchema 500-sample subset JSON
  - Video-MME official video chunks (20 zips -> 900 mp4)
  - LVBench official video chunks (14 zips -> 103 mp4)
  - LLaVA-NeXT third_party source checkout (pinned)
  - hf_home cache, env files, bootstrap_manifest.json

After every component finishes, the manifest is rewritten with verified counts
and timestamps, so re-running the orchestrator only does the missing pieces.

Usage:
  python scripts/bootstrap_assets.py plan
  python scripts/bootstrap_assets.py models --paper-models
  python scripts/bootstrap_assets.py videoespresso
  python scripts/bootstrap_assets.py all
  REVISE_ASSET_ROOT=/path/to/your/scratch/revise_paper_repro \\
    python scripts/bootstrap_assets.py all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path(
    os.environ.get(
        "REVISE_ASSET_ROOT", str(REPO_ROOT / "data" / "revise_assets")
    )
).expanduser()

# Pre-existing scratch trees the user may already maintain locally for other projects;
# we symlink them in when their layout matches what REVISE expects.
EXISTING_VIDEOESPRESSO_TEST = Path(
    "<SCRATCH>/VideoEspresso/VideoEspresso-Test"
)
EXISTING_VIDEOESPRESSO_TRAIN = Path(
    "<SCRATCH>/VideoEspresso/VideoEspresso_train_video"
)
EXISTING_NEXTQA_VIDEO = Path("<SCRATCH>/NExT-QA/NExTVideo")
EXISTING_OLD_MODELS_ROOT = Path(
    "<SCRATCH>/Sparse_Video_Understanding/revise_assets/models"
)

LLAVA_NEXT_REPO = "https://github.com/LLaVA-VL/LLaVA-NeXT.git"
LLAVA_NEXT_PIN = "df179663ae8b83207df100a1f7af24caec633ff9"

PAPER_MODEL_KEYS = (
    "qwen25_vl_3b",
    "qwen25_vl_7b",
    "qwen25_vl_72b",
    "qwen2_vl_7b",
    "internvl2_8b",
    "llava_ov_7b",
)


# ----------------------------------------------------------------------------
# Manifest + small helpers
# ----------------------------------------------------------------------------


@dataclass
class StepResult:
    name: str
    status: str  # "ok" | "skipped" | "failed"
    details: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _run(cmd: list[str], *, env: Optional[dict[str, str]] = None, cwd: Optional[Path] = None) -> tuple[int, str, str]:
    """Run a subprocess, capture output, return (rc, stdout, stderr)."""
    print(f"[bootstrap_assets] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode, proc.stdout, proc.stderr


def _count_mp4(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.mp4"))


def _ensure_symlink(target: Path, link: Path) -> None:
    """Create `link` as a symlink to `target`, replacing any existing entry."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        # If it's an empty dir skeleton from a previous extraction attempt,
        # remove it so the symlink replaces it cleanly.
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target)


def _read_manifest(asset_root: Path) -> dict[str, Any]:
    path = asset_root / "bootstrap_manifest.json"
    if not path.exists():
        return {"created": _ts(), "asset_root": str(asset_root), "steps": {}}
    return json.loads(path.read_text())


def _write_manifest(
    asset_root: Path,
    manifest: dict[str, Any],
    *,
    just_wrote: Optional[str] = None,
) -> None:
    """Read-merge-write with an fcntl exclusive lock + per-PID temp file.

    Concurrent Slurm jobs share `bootstrap_manifest.json`. To preserve every
    successful step, we (a) take an exclusive flock on a sibling `.lock` file
    so peers serialize on the write, (b) re-read on-disk inside the lock, and
    (c) merge only the step entry the caller just produced (`just_wrote`) over
    on-disk. The caller's stale in-memory copy of *other* steps is discarded,
    so a long-running job can't overwrite a peer's newer update.

    When `just_wrote` is None (e.g. the final bootstrap summary write), the
    caller's full `manifest['steps']` is merged (entries it touched this run).
    The lock + per-PID temp file also guarantee no two writers race on the
    same `.tmp` name.
    """
    import fcntl

    path = asset_root / "bootstrap_manifest.json"
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["asset_root"] = str(asset_root)
    manifest["updated"] = _ts()

    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        on_disk: dict[str, Any] = {}
        if path.exists():
            try:
                on_disk = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 - tolerate a partial write from a peer
                on_disk = {}

        merged_steps = dict(on_disk.get("steps", {}))
        own_steps = manifest.get("steps", {}) or {}
        if just_wrote is not None:
            if just_wrote in own_steps:
                merged_steps[just_wrote] = own_steps[just_wrote]
        else:
            merged_steps.update(own_steps)

        merged = dict(on_disk)
        # Top-level metadata (asset_root, updated) gets refreshed; we never
        # propagate stale top-level keys from in-memory `manifest`.
        merged["asset_root"] = manifest["asset_root"]
        merged["updated"] = manifest["updated"]
        merged["steps"] = merged_steps

        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str) + "\n")
        tmp.replace(path)
        # flock released on context exit.


def _record(manifest: dict[str, Any], result: StepResult) -> None:
    manifest.setdefault("steps", {})[result.name] = {
        "status": result.status,
        "elapsed_s": round(result.elapsed_s, 2),
        "updated": _ts(),
        **result.details,
    }


def _runner(name: str, manifest: dict[str, Any], asset_root: Path) -> Callable[[Callable[[], StepResult]], StepResult]:
    """Wrap a step closure with timing, manifest record, and exception capture."""

    def wrap(fn: Callable[[], StepResult]) -> StepResult:
        start = time.time()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - surface the cause in the manifest
            result = StepResult(name=name, status="failed", details={"error": repr(exc)})
        result.elapsed_s = time.time() - start
        _record(manifest, result)
        # Only this step's entry is merged into the on-disk manifest, so peers'
        # newer entries for other steps are never clobbered.
        _write_manifest(asset_root, manifest, just_wrote=name)
        marker = {"ok": "✓", "skipped": "·", "failed": "✗"}.get(result.status, "?")
        print(f"[bootstrap_assets] {marker} {name}: {result.status} ({result.elapsed_s:.1f}s) {result.details}", flush=True)
        return result

    return wrap


# ----------------------------------------------------------------------------
# Per-component bootstrap steps
# ----------------------------------------------------------------------------


def step_layout(asset_root: Path) -> StepResult:
    """Create the empty directory skeleton."""
    subdirs = [
        "models",
        "VideoEspresso/test_video",
        "VideoEspresso/train_video",
        "NExT-QA",
        "EgoSchema/videos",
        "Video-MME",
        "LVBench",
        "video_cache/videomme",
        "video_cache/lvbench",
        "third_party",
        "hf_home/hub",
        "hf_home/datasets",
        "hf_home/xet",
    ]
    for sub in subdirs:
        (asset_root / sub).mkdir(parents=True, exist_ok=True)
    return StepResult("layout", "ok", {"asset_root": str(asset_root)})


def step_models(asset_root: Path, *, models: list[str], dry_run: bool) -> StepResult:
    """Symlink models that already exist at the old root; download the rest.

    The previous-root `models/` dir survived earlier wipes intact (the cleanup
    only targets video files), so symlinking saves ~200G of re-download.
    """
    if not models:
        return StepResult("models", "skipped", {"reason": "no models selected"})

    spec_local = {
        "qwen25_vl_3b": "Qwen2.5-VL-3B-Instruct",
        "qwen25_vl_7b": "Qwen2.5-VL-7B-Instruct",
        "qwen25_vl_72b": "Qwen2.5-VL-72B-Instruct",
        "qwen2_vl_7b": "Qwen2-VL-7B-Instruct",
        "internvl2_8b": "InternVL2-8B",
        "llava_ov_7b": "LLaVA-OneVision-Qwen2-7B-OV",
    }

    models_root = asset_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    symlinked: list[str] = []
    missing: list[str] = []
    for key in models:
        local_name = spec_local.get(key)
        if not local_name:
            missing.append(key)
            continue
        new_dir = models_root / local_name
        if new_dir.exists() and any(new_dir.glob("*.safetensors")):
            continue  # already present (download or earlier symlink)
        existing = EXISTING_OLD_MODELS_ROOT / local_name
        if existing.exists() and any(existing.glob("*.safetensors")):
            _ensure_symlink(existing, new_dir)
            symlinked.append(key)
        else:
            missing.append(key)

    if not missing:
        return StepResult(
            "models",
            "ok",
            {"mode": "symlink-only", "symlinked": symlinked, "models": list(models)},
        )

    # Download what's missing via download_hf_models.py.
    env = os.environ.copy()
    env["REVISE_ASSET_ROOT"] = str(asset_root)
    env["HF_HOME"] = str(asset_root / "hf_home")
    env["HF_HUB_CACHE"] = str(asset_root / "hf_home" / "hub")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "repro" / "download_hf_models.py"),
        "--asset-root",
        str(asset_root),
    ]
    for name in missing:
        cmd.extend(["--model", name])
    if dry_run:
        cmd.append("--dry-run")

    rc, _, _ = _run(cmd, env=env)
    if rc != 0:
        return StepResult(
            "models",
            "failed",
            {"returncode": rc, "symlinked": symlinked, "downloaded_attempted": missing},
        )

    # Verify each requested model directory has shards present.
    present = {}
    for key in models:
        local_name = spec_local.get(key)
        if not local_name:
            present[key] = "unknown-key"
            continue
        mdir = models_root / local_name
        shards = len(list(mdir.glob("*.safetensors"))) if mdir.exists() else 0
        cfg_ok = (mdir / "config.json").exists()
        present[key] = {
            "path": str(mdir),
            "shards": shards,
            "config.json": cfg_ok,
            "symlink": mdir.is_symlink(),
        }
    return StepResult("models", "ok", {"symlinked": symlinked, "models": present})


def step_videoespresso_test(asset_root: Path, *, dry_run: bool) -> StepResult:
    """Symlink the user's pre-existing test set if present; else download.

    Always materializes `all_video/<category>/*.mp4` under
    `<asset_root>/VideoEspresso/test_video/`. bench_hard.json is copied/linked
    from the download (or from the existing layout if it carries one).
    """
    target = asset_root / "VideoEspresso" / "test_video"
    target.mkdir(parents=True, exist_ok=True)

    # Symlink-first.
    existing_all = EXISTING_VIDEOESPRESSO_TEST / "all_video"
    existing_count = _count_mp4(existing_all)
    if existing_count >= 1000:
        _ensure_symlink(existing_all, target / "all_video")
        existing_bench = EXISTING_VIDEOESPRESSO_TEST / "bench_hard.json"
        if existing_bench.exists():
            _ensure_symlink(existing_bench, target / "bench_hard.json")
        # Verify a sample resolves.
        try:
            rows = json.loads((target / "bench_hard.json").read_text())
            sample_rel = rows[0]["video_path"]
            sample_path = target / sample_rel
            if not sample_path.exists():
                # try with all_video prefix already in rel
                sample_path = target / "all_video" / sample_rel
            sample_ok = sample_path.exists()
        except Exception:  # noqa: BLE001
            sample_ok = False
        # A populated mp4 tree is not enough; the evaluation JSON has to
        # actually resolve. If it doesn't (missing/corrupt bench_hard.json or
        # video_path layout drift), don't pretend this step is ready.
        if not sample_ok:
            return StepResult(
                "videoespresso_test",
                "failed",
                {
                    "mode": "symlink",
                    "source": str(existing_all),
                    "mp4_count": existing_count,
                    "reason": "bench_hard.json sample does not resolve to a local mp4",
                },
            )
        return StepResult(
            "videoespresso_test",
            "ok",
            {
                "mode": "symlink",
                "source": str(existing_all),
                "mp4_count": existing_count,
                "sample_resolves": True,
            },
        )

    # Download fallback: reuse download_assets.py --videoespresso.
    if dry_run:
        return StepResult("videoespresso_test", "skipped", {"reason": "dry-run", "would_download": True})
    env = os.environ.copy()
    env["REVISE_ASSET_ROOT"] = str(asset_root)
    rc, _, _ = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "repro" / "download_assets.py"), "--videoespresso"],
        env=env,
    )
    if rc != 0:
        return StepResult("videoespresso_test", "failed", {"returncode": rc})

    # Extract all_video.zip if present.
    # `unzip -DD` skips restoring archived timestamps so extracted files
    # get current mtime — Quest scratch auto-purges by mtime and the
    # VideoEspresso zip carries 2024-vintage timestamps that would expire
    # within days otherwise. The post-extract `find -exec touch` is a
    # belt-and-suspenders refresh.
    zip_path = target / "all_video.zip"
    if zip_path.exists() and _count_mp4(target / "all_video") < 1000:
        env2 = os.environ.copy()
        env2["UNZIP_DISABLE_ZIPBOMB_DETECTION"] = "TRUE"
        rc2, _, _ = _run(
            ["unzip", "-o", "-q", "-DD", str(zip_path), "-x", "__MACOSX/*", "*.DS_Store"],
            env=env2,
            cwd=target,
        )
        if rc2 != 0:
            return StepResult("videoespresso_test", "failed", {"reason": "extract", "returncode": rc2})
        _run(["find", str(target / "all_video"), "-name", "*.mp4", "-exec", "touch", "{}", "+"])

    final_count = _count_mp4(target / "all_video")
    if final_count < 1000:
        return StepResult("videoespresso_test", "failed", {"reason": "mp4 count too low", "mp4_count": final_count})
    return StepResult("videoespresso_test", "ok", {"mode": "download", "mp4_count": final_count})


def step_videoespresso_train(asset_root: Path, *, dry_run: bool) -> StepResult:
    """Symlink train_video archives if present; else download.

    Selective per-experiment extraction stays the responsibility of
    `extract_videoespresso_split_zip_subset.py` invoked at paper-suite run
    time; the bootstrap only ensures the multi-part zip and metadata JSON
    are present.
    """
    target = asset_root / "VideoEspresso" / "train_video"
    target.mkdir(parents=True, exist_ok=True)

    if EXISTING_VIDEOESPRESSO_TRAIN.exists():
        # Symlink each split archive + the JSON. Keep all_video/ as a
        # writable dir on the new root (so selective extracts land there).
        n = 0
        for item in EXISTING_VIDEOESPRESSO_TRAIN.iterdir():
            if item.name == "all_video":
                continue
            link = target / item.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(item)
            n += 1
        # Ensure all_video dir exists locally.
        (target / "all_video").mkdir(exist_ok=True)
        return StepResult(
            "videoespresso_train",
            "ok",
            {"mode": "symlink", "linked_items": n, "source": str(EXISTING_VIDEOESPRESSO_TRAIN)},
        )

    # If step_videoespresso_test already triggered a fallback download, the
    # train archives landed in the same `download_assets.py --videoespresso`
    # call and we're done.
    if any(target.glob("*.z*")) or (target / "videoespresso_train_video.json").exists():
        return StepResult(
            "videoespresso_train",
            "ok",
            {"mode": "exists", "split_zip_parts": len(list(target.glob("*.z*")))},
        )
    if dry_run:
        return StepResult("videoespresso_train", "skipped", {"reason": "dry-run", "would_download": True})
    env = os.environ.copy()
    env["REVISE_ASSET_ROOT"] = str(asset_root)
    # download_assets.py --videoespresso fetches both the test/all_video.zip and
    # the multi-part train split archive into VideoEspresso/{test_video,train_video}/.
    rc, _, _ = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "repro" / "download_assets.py"),
            "--videoespresso",
            "--asset-root",
            str(asset_root),
        ],
        env=env,
    )
    if rc != 0:
        return StepResult("videoespresso_train", "failed", {"returncode": rc})
    return StepResult("videoespresso_train", "ok", {"mode": "download"})


def _ensure_nextqa_annotations(nextqa_root: Path, *, dry_run: bool) -> tuple[bool, dict[str, Any]]:
    """Make train/val/test CSV + map_vid_vidorID.json present. Returns (ok, info)."""
    ann_dir = nextqa_root / "nextqa"
    map_json = nextqa_root / "map_vid_vidorID.json"
    needed = [ann_dir / "train.csv", ann_dir / "val.csv", ann_dir / "test.csv", map_json]
    if all(p.exists() for p in needed):
        return True, {"annotations": "already-present"}
    if dry_run:
        return False, {"annotations": "skipped-dry-run", "missing": [str(p) for p in needed if not p.exists()]}
    repo_dir = nextqa_root / "_official_repo"
    if not repo_dir.exists():
        rc, _, err = _run(
            ["git", "clone", "--depth", "1", "https://github.com/doc-doc/NExT-QA.git", str(repo_dir)]
        )
        if rc != 0:
            return False, {"annotations": "git-clone-failed", "stderr": err[-400:]}
    src = repo_dir / "dataset" / "nextqa"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "val.csv", "test.csv", "map_vid_vidorID.json"):
        if (src / name).exists():
            shutil.copy2(src / name, ann_dir / name)
    if (ann_dir / "map_vid_vidorID.json").exists():
        shutil.copy2(ann_dir / "map_vid_vidorID.json", map_json)
    missing = [p for p in needed if not p.exists()]
    if missing:
        return False, {"annotations": "still-missing-after-clone", "missing": [str(p) for p in missing]}
    return True, {"annotations": "cloned"}


def step_nextqa(asset_root: Path, *, dry_run: bool) -> StepResult:
    """Ensure annotations + videos. Symlink-first for videos when an existing
    populated tree is available; otherwise download NExTVideo.zip (~24G) and
    extract. Annotations (train/val/test CSV + map_vid_vidorID.json) are
    always materialized — symlinking only the videos leaves runs unrunnable.
    """
    nextqa_root = asset_root / "NExT-QA"
    nextqa_root.mkdir(parents=True, exist_ok=True)

    ann_ok, ann_info = _ensure_nextqa_annotations(nextqa_root, dry_run=dry_run)
    if not ann_ok and not dry_run:
        return StepResult("nextqa_videos", "failed", {"stage": "annotations", **ann_info})

    # Videos: symlink-first if the user has a populated pre-existing NExT-QA tree.
    existing_count = _count_mp4(EXISTING_NEXTQA_VIDEO)
    if existing_count >= 5000:
        _ensure_symlink(EXISTING_NEXTQA_VIDEO, nextqa_root / "NExTVideo")
        return StepResult(
            "nextqa_videos",
            "ok",
            {
                "mode": "symlink",
                "source": str(EXISTING_NEXTQA_VIDEO),
                "mp4_count": existing_count,
                **ann_info,
            },
        )

    # Download path: reuse download_assets.py --nextqa-raw-hf for the videos.
    # (Annotations are already in place above.)
    if dry_run:
        return StepResult(
            "nextqa_videos",
            "skipped",
            {"reason": "dry-run", "would_download": True, "zip": str(nextqa_root / "NExTVideo.zip"), **ann_info},
        )

    env = os.environ.copy()
    env["REVISE_ASSET_ROOT"] = str(asset_root)
    env["HF_HOME"] = str(asset_root / "hf_home")
    env["HF_HUB_CACHE"] = str(asset_root / "hf_home" / "hub")
    rc, _, _ = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "repro" / "download_assets.py"),
            "--nextqa-raw-hf",
        ],
        env=env,
    )
    if rc != 0:
        return StepResult("nextqa_videos", "failed", {"returncode": rc, "stage": "download"})

    # Extract via direct unzip (with zipbomb workaround).
    zip_path = nextqa_root / "NExTVideo.zip"
    if not zip_path.exists():
        return StepResult("nextqa_videos", "failed", {"reason": "zip missing after download"})

    # `unzip -DD` skips restoring archived timestamps so extracted files
    # get current mtime — Quest scratch auto-purges by mtime and the
    # NExTVideo zip carries 2024-vintage timestamps that would expire
    # within days otherwise. The post-extract `find -exec touch` is a
    # belt-and-suspenders refresh.
    env2 = os.environ.copy()
    env2["UNZIP_DISABLE_ZIPBOMB_DETECTION"] = "TRUE"
    rc2, _, _ = _run(
        ["unzip", "-o", "-q", "-DD", str(zip_path), "-x", "__MACOSX/*", "*.DS_Store"],
        env=env2,
        cwd=nextqa_root,
    )
    if rc2 != 0:
        return StepResult("nextqa_videos", "failed", {"reason": "extract", "returncode": rc2})
    _run(["find", str(nextqa_root / "NExTVideo"), "-name", "*.mp4", "-exec", "touch", "{}", "+"])

    final_count = _count_mp4(nextqa_root / "NExTVideo")
    if final_count < 5000:
        return StepResult(
            "nextqa_videos",
            "failed",
            {"reason": "mp4 count too low after extract", "mp4_count": final_count},
        )
    return StepResult("nextqa_videos", "ok", {"mode": "download", "mp4_count": final_count})


def step_egoschema(asset_root: Path, *, dry_run: bool) -> StepResult:
    """Materialize the 500-sample EgoSchema subset JSON."""
    out_json = asset_root / "EgoSchema" / "pnp_subset_500.json"
    if out_json.exists() and out_json.stat().st_size > 1000:
        return StepResult("egoschema_subset", "ok", {"path": str(out_json), "mode": "exists"})
    if dry_run:
        return StepResult("egoschema_subset", "skipped", {"reason": "dry-run"})
    env = os.environ.copy()
    env["REVISE_ASSET_ROOT"] = str(asset_root)
    env["HF_HOME"] = str(asset_root / "hf_home")
    rc, _, _ = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "repro" / "prepare_egoschema_subset.py"),
            "--out-json",
            str(out_json),
            "--asset-root",
            str(asset_root),
        ],
        env=env,
    )
    if rc != 0:
        return StepResult("egoschema_subset", "failed", {"returncode": rc})
    return StepResult("egoschema_subset", "ok", {"path": str(out_json), "mode": "download"})


def step_videomme(asset_root: Path, *, dry_run: bool) -> StepResult:
    """Download Video-MME official chunked HF zips (900 mp4 total)."""
    cache_dir = asset_root / "video_cache" / "videomme"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = _count_mp4(cache_dir)
    if existing >= 900:
        return StepResult("videomme", "ok", {"mp4_count": existing, "mode": "already-cached"})
    if dry_run:
        return StepResult(
            "videomme", "skipped", {"reason": "dry-run", "current_mp4": existing, "would_download_to": str(cache_dir)},
        )
    env = os.environ.copy()
    env["REVISE_ASSET_ROOT"] = str(asset_root)
    env["HF_HOME"] = str(asset_root / "hf_home")
    rc, _, _ = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "repro" / "download_videomme_official_videos.py"),
            "--asset-root",
            str(asset_root),
            # Downloader appends `videomme/` itself; pass the parent.
            "--video-cache-dir",
            str(cache_dir.parent),
            "--delete-zip-after-extract",
        ],
        env=env,
    )
    if rc != 0:
        return StepResult("videomme", "failed", {"returncode": rc})
    final = _count_mp4(cache_dir)
    return StepResult("videomme", "ok" if final >= 900 else "failed", {"mp4_count": final})


def step_lvbench(asset_root: Path, *, dry_run: bool) -> StepResult:
    """Download LVBench official HF chunks (103 mp4 total)."""
    cache_dir = asset_root / "video_cache" / "lvbench"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = _count_mp4(cache_dir)
    if existing >= 103:
        return StepResult("lvbench", "ok", {"mp4_count": existing, "mode": "already-cached"})
    if dry_run:
        return StepResult("lvbench", "skipped", {"reason": "dry-run", "current_mp4": existing})
    env = os.environ.copy()
    env["REVISE_ASSET_ROOT"] = str(asset_root)
    env["HF_HOME"] = str(asset_root / "hf_home")
    rc, _, _ = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "repro" / "download_lvbench_official_videos.py"),
            "--asset-root",
            str(asset_root),
            # Downloader appends `lvbench/` itself; pass the parent.
            "--video-cache-dir",
            str(cache_dir.parent),
            "--delete-zip-after-extract",
        ],
        env=env,
    )
    if rc != 0:
        return StepResult("lvbench", "failed", {"returncode": rc})
    final = _count_mp4(cache_dir)
    return StepResult("lvbench", "ok" if final >= 103 else "failed", {"mp4_count": final})


def step_llava_next(asset_root: Path, *, dry_run: bool) -> StepResult:
    """Pin a LLaVA-NeXT source checkout for LLaVA-OV-7B HF inference.

    Reproducibility: HEAD must equal `LLAVA_NEXT_PIN`. If an existing checkout
    is at a different commit (e.g. a manual checkout or partial clone), reset
    it to the pin rather than silently accepting it.
    """
    target = asset_root / "third_party" / "LLaVA-NeXT"
    if (target / ".git").exists():
        rc, head_out, _ = _run(["git", "-C", str(target), "rev-parse", "HEAD"])
        head = head_out.strip()
        if rc == 0 and head == LLAVA_NEXT_PIN:
            return StepResult("llava_next", "ok", {"mode": "exists-pinned", "path": str(target), "head": head})
        if dry_run:
            return StepResult(
                "llava_next",
                "skipped",
                {"reason": "dry-run", "head": head, "expected_pin": LLAVA_NEXT_PIN},
            )
        # Fetch + hard-reset to the pin. Tolerant of detached HEAD.
        _run(["git", "-C", str(target), "fetch", "--depth", "50", "origin", LLAVA_NEXT_PIN])
        rc2, _, err = _run(["git", "-C", str(target), "reset", "--hard", LLAVA_NEXT_PIN])
        if rc2 != 0:
            return StepResult(
                "llava_next",
                "failed",
                {"stage": "reset-to-pin", "returncode": rc2, "stderr": err[-400:], "had_head": head},
            )
        return StepResult(
            "llava_next",
            "ok",
            {"mode": "reset-to-pin", "path": str(target), "pin": LLAVA_NEXT_PIN, "previous_head": head},
        )
    if dry_run:
        return StepResult("llava_next", "skipped", {"reason": "dry-run", "would_clone": LLAVA_NEXT_REPO})
    target.parent.mkdir(parents=True, exist_ok=True)
    rc, _, _ = _run(["git", "clone", LLAVA_NEXT_REPO, str(target)])
    if rc != 0:
        return StepResult("llava_next", "failed", {"returncode": rc, "stage": "clone"})
    rc, _, _ = _run(["git", "-C", str(target), "checkout", LLAVA_NEXT_PIN])
    if rc != 0:
        return StepResult("llava_next", "failed", {"returncode": rc, "stage": "checkout"})
    return StepResult("llava_next", "ok", {"mode": "fresh-clone", "path": str(target), "pin": LLAVA_NEXT_PIN})


def step_env(asset_root: Path) -> StepResult:
    """Write revise_env.sh and revise_models_env.sh from the populated tree."""
    exports: dict[str, str] = {
        "REVISE_ASSET_ROOT": str(asset_root),
        "REVISE_NEXTQA_ROOT": str(asset_root / "NExT-QA"),
        "REVISE_NEXTQA_VIDEO_ROOT": str(asset_root / "NExT-QA" / "NExTVideo"),
        "REVISE_NEXTQA_MAP_JSON": str(asset_root / "NExT-QA" / "map_vid_vidorID.json"),
        "REVISE_NEXTQA_TRAIN_CSV": str(asset_root / "NExT-QA" / "nextqa" / "train.csv"),
        "REVISE_NEXTQA_VAL_CSV": str(asset_root / "NExT-QA" / "nextqa" / "val.csv"),
        "REVISE_VIDEO_CACHE_DIR": str(asset_root / "video_cache"),
        "HF_HOME": str(asset_root / "hf_home"),
        "HF_HUB_CACHE": str(asset_root / "hf_home" / "hub"),
        "HF_DATASETS_CACHE": str(asset_root / "hf_home" / "datasets"),
        "HF_XET_CACHE": str(asset_root / "hf_home" / "xet"),
    }
    env_path = asset_root / "revise_env.sh"
    lines = ["# Source this file before running scripts/doctor.py or paper_suite.py."]
    for k, v in sorted(exports.items()):
        lines.append(f"export {k}={json.dumps(v)}")
    env_path.write_text("\n".join(lines) + "\n")

    # Models env: include only those that materialize on disk.
    spec_map = {
        "qwen25_vl_3b": ("Qwen2.5-VL-3B-Instruct", "REVISE_QWEN25_VL_3B_PATH"),
        "qwen25_vl_7b": ("Qwen2.5-VL-7B-Instruct", "REVISE_QWEN25_VL_7B_PATH"),
        "qwen25_vl_72b": ("Qwen2.5-VL-72B-Instruct", "REVISE_QWEN25_VL_72B_PATH"),
        "qwen2_vl_7b": ("Qwen2-VL-7B-Instruct", "REVISE_QWEN2_VL_7B_PATH"),
        "internvl2_8b": ("InternVL2-8B", "REVISE_INTERNVL2_8B_PATH"),
        "llava_ov_7b": ("LLaVA-OneVision-Qwen2-7B-OV", "REVISE_LLAVA_OV_7B_PATH"),
    }
    model_exports: dict[str, str] = {}
    for _, (local_name, env_var) in spec_map.items():
        d = asset_root / "models" / local_name
        if d.exists() and any(d.glob("*.safetensors")):
            model_exports[env_var] = str(d)
    if (asset_root / "third_party" / "LLaVA-NeXT").exists():
        model_exports["REVISE_LLAVA_NEXT_PATH"] = str(asset_root / "third_party" / "LLaVA-NeXT")
    models_env_path = asset_root / "revise_models_env.sh"
    mlines = ["# Source this file to register exact local model snapshots for paper-row reruns."]
    for k, v in sorted(model_exports.items()):
        mlines.append(f"export {k}={json.dumps(v)}")
    models_env_path.write_text("\n".join(mlines) + "\n")

    return StepResult(
        "env",
        "ok",
        {"revise_env": str(env_path), "models_env": str(models_env_path), "models_registered": len(model_exports)},
    )


# ----------------------------------------------------------------------------
# Plan / CLI
# ----------------------------------------------------------------------------


ALL_STEP_NAMES = [
    "layout",
    "models",
    "videoespresso_test",
    "videoespresso_train",
    "nextqa_videos",
    "egoschema_subset",
    "videomme",
    "lvbench",
    "llava_next",
    "env",
]


def cmd_plan(asset_root: Path, *, models: list[str]) -> int:
    print(f"# Bootstrap plan for {asset_root}\n")
    print(f"Asset root exists: {asset_root.exists()}")
    print(f"Existing VE test (1364 mp4 expected): {_count_mp4(EXISTING_VIDEOESPRESSO_TEST / 'all_video')} mp4 at {EXISTING_VIDEOESPRESSO_TEST}")
    print(f"Existing VE train dir: {EXISTING_VIDEOESPRESSO_TRAIN.exists()}")
    print(f"Existing NExT-QA videos: {_count_mp4(EXISTING_NEXTQA_VIDEO)} mp4 at {EXISTING_NEXTQA_VIDEO}")
    print(f"\nSteps that will run (in order): {', '.join(ALL_STEP_NAMES)}")
    if models:
        print(f"Models: {', '.join(models)}")
    print()
    print("Re-run with one of:")
    print("  bootstrap_assets.py all          # everything")
    print("  bootstrap_assets.py models       # just HF models")
    print("  bootstrap_assets.py videoespresso # test + train")
    print("  bootstrap_assets.py nextqa")
    print("  bootstrap_assets.py videomme")
    print("  bootstrap_assets.py lvbench")
    return 0


def cmd_run(asset_root: Path, *, only: Optional[list[str]], models: list[str], dry_run: bool) -> int:
    asset_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(asset_root)

    selected = set(only or ALL_STEP_NAMES)
    rc = 0
    ran_steps: list[str] = []

    def maybe(name: str, fn: Callable[[], StepResult]) -> Optional[StepResult]:
        if name not in selected:
            return None
        result = _runner(name, manifest, asset_root)(fn)
        ran_steps.append(name)
        return result

    maybe("layout", lambda: step_layout(asset_root))
    maybe("models", lambda: step_models(asset_root, models=models, dry_run=dry_run))
    maybe("videoespresso_test", lambda: step_videoespresso_test(asset_root, dry_run=dry_run))
    maybe("videoespresso_train", lambda: step_videoespresso_train(asset_root, dry_run=dry_run))
    maybe("nextqa_videos", lambda: step_nextqa(asset_root, dry_run=dry_run))
    maybe("egoschema_subset", lambda: step_egoschema(asset_root, dry_run=dry_run))
    maybe("videomme", lambda: step_videomme(asset_root, dry_run=dry_run))
    maybe("lvbench", lambda: step_lvbench(asset_root, dry_run=dry_run))
    maybe("llava_next", lambda: step_llava_next(asset_root, dry_run=dry_run))
    maybe("env", lambda: step_env(asset_root))

    # Aggregate exit code based ONLY on steps this invocation actually ran.
    # Older stale `failed` entries from previous invocations (which are now
    # superseded by later `ok` entries on disk) must not count against this run.
    steps = manifest.get("steps", {})
    for name in ran_steps:
        info = steps.get(name, {})
        if info.get("status") == "failed":
            print(f"[bootstrap_assets] FAILED step: {name} :: {info}", file=sys.stderr)
            rc = 1

    print(f"\n[bootstrap_assets] Manifest: {asset_root / 'bootstrap_manifest.json'}")
    return rc


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap a REVISE asset root.")
    ap.add_argument("command", choices=["plan"] + ALL_STEP_NAMES + ["videoespresso", "all"])
    ap.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeatable. HF model key (qwen25_vl_3b, qwen25_vl_7b, qwen25_vl_72b, qwen2_vl_7b, internvl2_8b, llava_ov_7b). Defaults to all paper models.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    asset_root = Path(args.asset_root).expanduser().resolve()
    models = args.model or list(PAPER_MODEL_KEYS)

    if args.command == "plan":
        return cmd_plan(asset_root, models=models)

    if args.command == "all":
        only = None
    elif args.command == "videoespresso":
        only = ["layout", "videoespresso_test", "videoespresso_train", "env"]
    else:
        only = ["layout", args.command, "env"]

    return cmd_run(asset_root, only=only, models=models, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
