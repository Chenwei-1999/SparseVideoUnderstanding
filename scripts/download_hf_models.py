#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    local_name: str
    env_var: str
    paper_rows: tuple[str, ...]


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen25_vl_3b": ModelSpec(
        key="qwen25_vl_3b",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        local_name="Qwen2.5-VL-3B-Instruct",
        env_var="REVISE_QWEN25_VL_3B_PATH",
        paper_rows=("training backbone",),
    ),
    "qwen25_vl_7b": ModelSpec(
        key="qwen25_vl_7b",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_name="Qwen2.5-VL-7B-Instruct",
        env_var="REVISE_QWEN25_VL_7B_PATH",
        paper_rows=("VideoEspresso ablation", "teacher fallback"),
    ),
    "qwen25_vl_72b": ModelSpec(
        key="qwen25_vl_72b",
        repo_id="Qwen/Qwen2.5-VL-72B-Instruct",
        local_name="Qwen2.5-VL-72B-Instruct",
        env_var="REVISE_QWEN25_VL_72B_PATH",
        paper_rows=("SFT teacher (72B distillation, GPT-4o substitute)",),
    ),
    "qwen35_4b": ModelSpec(
        key="qwen35_4b",
        repo_id="Qwen/Qwen3.5-4B",
        local_name="Qwen3.5-4B",
        env_var="REVISE_QWEN35_4B_PATH",
        paper_rows=("experimental NExT-QA ablation; not Table 4 comparable",),
    ),
    "qwen2_vl_7b": ModelSpec(
        key="qwen2_vl_7b",
        repo_id="Qwen/Qwen2-VL-7B-Instruct",
        local_name="Qwen2-VL-7B-Instruct",
        env_var="REVISE_QWEN2_VL_7B_PATH",
        paper_rows=("VideoEspresso Qwen2-VL + ReViSe",),
    ),
    "internvl2_8b": ModelSpec(
        key="internvl2_8b",
        repo_id="OpenGVLab/InternVL2-8B",
        local_name="InternVL2-8B",
        env_var="REVISE_INTERNVL2_8B_PATH",
        paper_rows=("VideoEspresso InternVL2 + ReViSe",),
    ),
    "llava_ov_7b": ModelSpec(
        key="llava_ov_7b",
        repo_id="lmms-lab/llava-onevision-qwen2-7b-ov",
        local_name="LLaVA-OneVision-Qwen2-7B-OV",
        env_var="REVISE_LLAVA_OV_7B_PATH",
        paper_rows=("Video-MME/LVBench LLaVA-OV-7B + ReViSe",),
    ),
}

PAPER_MODEL_KEYS = ("qwen2_vl_7b", "internvl2_8b", "llava_ov_7b")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_env(path: Path, exports: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Source this file to register exact local model snapshots for paper-row reruns."]
    for key, value in sorted(exports.items()):
        lines.append(f"export {key}={json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _existing_model_exports(model_root: Path) -> dict[str, str]:
    exports: dict[str, str] = {}
    for spec in MODEL_SPECS.values():
        local_dir = model_root / spec.local_name
        if local_dir.exists():
            exports[spec.env_var] = str(local_dir)
    return exports


def _selected_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []
    if args.all:
        keys.extend(MODEL_SPECS)
    if args.paper_models:
        keys.extend(PAPER_MODEL_KEYS)
    keys.extend(args.model or [])
    if not keys:
        raise SystemExit("Specify --model, --paper-models, or --all.")
    unknown = sorted({key for key in keys if key not in MODEL_SPECS})
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")
    return list(dict.fromkeys(keys))


def _model_size_gb(repo_id: str) -> tuple[str | None, int | None, float | None]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id=repo_id, files_metadata=True)
    except Exception:
        return None, None, None
    total = 0
    files = 0
    for sibling in info.siblings or []:
        size = getattr(sibling, "size", None) or 0
        total += int(size)
        files += 1
    sha = getattr(info, "sha", None)
    return sha, files, total / (1024**3)


def _snapshot_download(spec: ModelSpec, local_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    sha, files, total_gb = _model_size_gb(spec.repo_id)
    entry: dict[str, Any] = {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "local_dir": str(local_dir),
        "env_var": spec.env_var,
        "paper_rows": list(spec.paper_rows),
        "sha": sha,
        "files": files,
        "total_gb": total_gb,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        entry["download"] = "dry_run"
        return entry

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    entry["download"] = snapshot_download(
        repo_id=spec.repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Download exact HF model snapshots used by REVISE paper rows.")
    parser.add_argument("--model", action="append", choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--paper-models",
        action="store_true",
        help="Download Qwen2-VL-7B, InternVL2-8B, and LLaVA-OV-7B.",
    )
    parser.add_argument("--all", action="store_true", help="Download every known local model snapshot.")
    parser.add_argument(
        "--asset-root",
        default=os.getenv("REVISE_ASSET_ROOT", str(REPO_ROOT / "data" / "revise_assets")),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    model_root = asset_root / "models"
    os.environ.setdefault("HF_HOME", str(asset_root / ".hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(asset_root / ".hf_home" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(asset_root / ".hf_home" / "datasets"))
    os.environ.setdefault("HF_XET_CACHE", str(asset_root / ".hf_home" / "xet"))

    report: dict[str, Any] = {
        "asset_root": str(asset_root),
        "model_root": str(model_root),
        "dry_run": bool(args.dry_run),
        "models": [],
    }
    exports: dict[str, str] = _existing_model_exports(model_root)
    for key in _selected_keys(args):
        spec = MODEL_SPECS[key]
        local_dir = model_root / spec.local_name
        entry = _snapshot_download(spec, local_dir, dry_run=bool(args.dry_run))
        report["models"].append(entry)
        exports[spec.env_var] = str(local_dir)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    env_path = asset_root / "revise_models_env.sh"
    _write_env(env_path, exports)
    report["env_file"] = str(env_path)
    manifest = Path(args.manifest).expanduser() if args.manifest else asset_root / "model_download_manifest.json"
    _write_json(manifest, report)
    print(json.dumps({"manifest": str(manifest), "env_file": str(env_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
