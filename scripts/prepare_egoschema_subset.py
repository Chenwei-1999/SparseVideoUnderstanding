#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = REPO_ROOT / "data" / "revise_assets"
DEFAULT_REPO = "lmms-lab/egoschema"
DEFAULT_CONFIG = "Subset"


def _answer_to_letter(answer: Any, num_options: int) -> str:
    raw = str(answer).strip()
    if len(raw) == 1 and raw.upper().isalpha():
        idx = ord(raw.upper()) - ord("A")
        if 0 <= idx < num_options:
            return raw.upper()
    try:
        idx = int(raw)
    except Exception as exc:
        raise ValueError(f"invalid EgoSchema answer: {answer!r}") from exc
    if not (0 <= idx < num_options):
        raise ValueError(f"answer index {idx} outside option range 0..{num_options - 1}")
    return chr(ord("A") + idx)


def canonicalize_egoschema_row(row: dict[str, Any]) -> dict[str, Any]:
    options = row.get("option") or row.get("options") or []
    if not isinstance(options, list) or not options:
        raise ValueError("EgoSchema row missing option/options list")
    video_idx = str(row.get("video_idx") or row.get("video_id") or "").strip()
    if not video_idx:
        raise ValueError("EgoSchema row missing video_idx")
    video_path = video_idx if video_idx.endswith(".mp4") else f"{video_idx}.mp4"
    return {
        "question_idx": str(row.get("question_idx") or row.get("qid") or "").strip(),
        "question": str(row.get("question") or "").strip(),
        "options": [str(opt).strip() for opt in options],
        "correct_answer": _answer_to_letter(row.get("answer"), len(options)),
        "video_path": video_path,
    }


def load_hf_subset(repo_id: str, config: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is required to materialize EgoSchema subset") from exc
    ds = load_dataset(repo_id, config, split="test")
    return [canonicalize_egoschema_row(dict(row)) for row in ds]


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the public EgoSchema 500-answer subset JSON.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--asset-root",
        default=os.getenv("REVISE_ASSET_ROOT", str(DEFAULT_ASSET_ROOT)),
        help="Ignored asset root containing EgoSchema/ and HF caches.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Output JSON path. Defaults to <asset-root>/EgoSchema/pnp_subset_500.json.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(asset_root / ".hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(asset_root / ".hf_home" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(asset_root / ".hf_home" / "datasets"))
    os.environ.setdefault("HF_XET_CACHE", str(asset_root / ".hf_home" / "xet"))

    out_json = Path(args.out_json).expanduser() if args.out_json else asset_root / "EgoSchema" / "pnp_subset_500.json"
    rows = load_hf_subset(str(args.repo_id), str(args.config))
    report = {
        "repo_id": str(args.repo_id),
        "config": str(args.config),
        "rows": len(rows),
        "out_json": str(out_json),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
