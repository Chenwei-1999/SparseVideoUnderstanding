#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repro.common import discover_assets
from examples.revise.pnp_utils import normalize_video_id, resolve_nextqa_video_path


def _ok(value: str | None) -> str:
    return "ok" if value else "missing"


def _probe_nextqa_sample(nextqa: dict) -> dict:
    required = [nextqa.get("video_root"), nextqa.get("map_json"), nextqa.get("val_csv")]
    if not all(required):
        return {"ok": False, "reason": "missing required paths"}

    with open(nextqa["map_json"], "r", encoding="utf-8") as f:
        video_map = {str(k): v for k, v in json.load(f).items()}

    checked = 0
    with open(nextqa["val_csv"], "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            checked += 1
            video_id = normalize_video_id(row.get("video", ""))
            rel = video_map.get(video_id)
            if rel is None:
                continue
            video_path = resolve_nextqa_video_path(nextqa["video_root"], str(rel), video_id)
            if video_path:
                return {"ok": True, "video_id": video_id, "path": video_path, "checked": checked}
            if checked >= 50:
                break
    return {"ok": False, "reason": "no resolvable validation video among first rows", "checked": checked}


def build_report() -> dict:
    assets = discover_assets()
    blockers: list[str] = []
    warnings: list[str] = []

    packages = assets["packages"]
    required_packages = ["torch", "transformers", "vllm", "decord", "datasets", "hydra-core", "ray"]
    for pkg in required_packages:
        if not packages.get(pkg):
            blockers.append(f"Missing required package: {pkg}")

    if packages.get("vllm") and packages.get("sglang"):
        warnings.append(
            "Both vLLM and SGLang are installed in one environment. Keep separate envs for reproducible runs."
        )

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    pip_check_output = "\n".join(part for part in (pip_check.stdout, pip_check.stderr) if part)
    pip_check_lines = [line.strip() for line in pip_check_output.splitlines() if line.strip()]
    if pip_check.returncode != 0:
        warnings.append("`python -m pip check` reported dependency conflicts in this environment.")

    nextqa = assets["datasets"]["nextqa"]
    for key in ("video_root", "map_json", "train_csv", "val_csv"):
        if not nextqa.get(key):
            blockers.append(f"NExT-QA asset missing: {key}")
    if all(nextqa.get(key) for key in ("video_root", "map_json", "val_csv")):
        nextqa["val_probe"] = nextqa.get("val_probe") or _probe_nextqa_sample(nextqa)
        if not nextqa["val_probe"].get("ok"):
            blockers.append(f"NExT-QA validation videos are not resolvable: {nextqa['val_probe'].get('reason')}")
    if nextqa.get("train_csv") and nextqa.get("train_probe") and not nextqa["train_probe"].get("ok"):
        warnings.append(
            "NExT-QA training videos are not resolvable; SFT/GRPO training requires the official raw train videos."
        )

    models = assets["models"]
    remote_api = assets["remote_api"]
    if not models.get("qwen25_vl_3b") and not (remote_api.get("base_url") and remote_api.get("model_id")):
        blockers.append("No local Qwen2.5-VL-3B path and no remote OpenAI-compatible API configured.")

    videoespresso = assets["datasets"]["videoespresso"]
    if not videoespresso.get("test_json") or not videoespresso.get("test_video_root"):
        blockers.append("VideoEspresso test set not found.")
    if not videoespresso.get("train_video_json"):
        blockers.append("VideoEspresso open-ended train JSON not found.")
    elif not videoespresso["mc_train_probe"].get("multiple_choice"):
        warnings.append(
            "VideoEspresso MC train JSON not found; reproduction will synthesize one from the open-ended public train file."
        )

    egoschema = assets["datasets"]["egoschema"]
    if not packages.get("datasets"):
        blockers.append("EgoSchema HF fallback requires the datasets package.")
    elif not egoschema.get("video_root") or not egoschema.get("json"):
        warnings.append("EgoSchema local JSON/video root not found; runs will fall back to Hugging Face and download videos on demand.")

    return {
        "assets": assets,
        "blockers": blockers,
        "warnings": warnings,
        "pip_check": {
            "ok": pip_check.returncode == 0,
            "lines": pip_check_lines,
        },
    }


def print_text(report: dict) -> None:
    assets = report["assets"]
    blockers = report["blockers"]
    warnings = report.get("warnings", [])
    packages = assets["packages"]

    print("Environment")
    print(f"- python: {assets['python']['path']} ({assets['python']['version']})")
    print(f"- gpu: {assets['gpu']['raw'] or 'not detected'}")

    print("\nPackages")
    for pkg in ["torch", "transformers", "vllm", "sglang", "decord", "datasets", "hydra-core", "ray", "wandb", "scikit-learn"]:
        print(f"- {pkg}: {packages.get(pkg) or 'missing'}")

    nextqa = assets["datasets"]["nextqa"]
    videoespresso = assets["datasets"]["videoespresso"]
    egoschema = assets["datasets"]["egoschema"]
    cache = assets["datasets"]["video_cache"]

    print("\nDatasets")
    print(f"- NExT-QA: {_ok(nextqa.get('video_root'))} | {nextqa.get('video_root') or 'unset'}")
    print(f"- NExT-QA map: {_ok(nextqa.get('map_json'))} | {nextqa.get('map_json') or 'unset'}")
    print(f"- NExT-QA train csv: {_ok(nextqa.get('train_csv'))} | {nextqa.get('train_csv') or 'unset'}")
    print(f"- NExT-QA val csv: {_ok(nextqa.get('val_csv'))} | {nextqa.get('val_csv') or 'unset'}")
    val_probe = nextqa.get("val_probe") or {}
    train_probe = nextqa.get("train_probe") or {}
    print(
        f"- NExT-QA val sample probe: "
        f"{'ok' if val_probe.get('ok') else 'missing'} | {val_probe.get('path') or val_probe.get('reason') or 'unset'}"
    )
    print(
        f"- NExT-QA train sample probe: "
        f"{'ok' if train_probe.get('ok') else 'missing'} | {train_probe.get('path') or train_probe.get('reason') or 'unset'}"
    )
    print(f"- VideoEspresso eval: {_ok(videoespresso.get('test_json'))} | {videoespresso.get('test_json') or 'unset'}")
    print(
        f"- VideoEspresso public train MC: "
        f"{'yes' if videoespresso['public_train_probe']['multiple_choice'] else 'no'} "
        f"({videoespresso['public_train_probe']['reason']})"
    )
    print(
        f"- VideoEspresso MC train override: "
        f"{'yes' if videoespresso['mc_train_probe']['multiple_choice'] else 'no'} "
        f"({videoespresso['mc_train_probe']['reason']})"
    )
    print(f"- EgoSchema: {_ok(egoschema.get('video_root'))} | {egoschema.get('video_root') or 'unset'}")
    print(f"- EgoSchema json: {_ok(egoschema.get('json'))} | {egoschema.get('json') or 'unset'}")
    print(f"- Video cache root: {_ok(cache.get('root'))} | {cache.get('root') or 'unset'}")

    print("\nModels / API")
    print(f"- local Qwen2.5-VL-3B: {_ok(assets['models'].get('qwen25_vl_3b'))} | {assets['models'].get('qwen25_vl_3b') or 'unset'}")
    print(f"- local Qwen2.5-VL-7B: {_ok(assets['models'].get('qwen25_vl_7b'))} | {assets['models'].get('qwen25_vl_7b') or 'unset'}")
    print(f"- SFT teacher Qwen2.5-VL-72B: {_ok(assets['models'].get('qwen25_vl_72b'))} | {assets['models'].get('qwen25_vl_72b') or 'unset'}")
    print(f"- exact Qwen2-VL-7B: {_ok(assets['models'].get('qwen2_vl_7b'))} | {assets['models'].get('qwen2_vl_7b') or 'unset'}")
    print(f"- exact InternVL2-8B: {_ok(assets['models'].get('internvl2_8b'))} | {assets['models'].get('internvl2_8b') or 'unset'}")
    print(f"- exact LLaVA-OV-7B: {_ok(assets['models'].get('llava_ov_7b'))} | {assets['models'].get('llava_ov_7b') or 'unset'}")
    print(f"- LLaVA-NeXT source: {_ok(assets['models'].get('llava_next_path'))} | {assets['models'].get('llava_next_path') or 'unset'}")
    print(f"- override local model: {_ok(assets['models'].get('local_model'))} | {assets['models'].get('local_model') or 'unset'}")
    print(f"- override local model id: {assets['models'].get('local_model_id') or 'unset'}")
    print(f"- remote API base_url: {assets['remote_api']['base_url'] or 'unset'}")
    print(f"- remote API model_id: {assets['remote_api']['model_id'] or 'unset'}")
    print(f"- remote API key present: {'yes' if assets['remote_api']['api_key_present'] else 'no'}")

    print("\nCompatibility")
    pip_check = report.get("pip_check", {})
    print(f"- pip check: {'ok' if pip_check.get('ok') else 'issues'}")
    for line in pip_check.get("lines", [])[:12]:
        print(f"  {line}")

    print("\nWarnings")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("- none")

    print("\nBlockers")
    if blockers:
        for blocker in blockers:
            print(f"- {blocker}")
    else:
        print("- none")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check local environment and assets for paper reproduction.")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if blockers are present.")
    args = ap.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text(report)

    return 1 if args.strict and report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
