#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import REPO_ROOT, discover_assets
from scripts.check_video_cache_coverage import _coverage, _dataset_split


ExperimentBuilder = Callable[[dict, bool, Path], Union[list, str]]
ExperimentChecker = Callable[[dict, bool], list]


def _python_bin() -> str:
    return os.getenv("REVISE_PYTHON", sys.executable)


def _model_endpoint_args(assets: dict, *, prefer_7b: bool, port: int, server_log: Path | None = None) -> list[str]:
    remote = assets["remote_api"]
    if remote.get("base_url") and remote.get("model_id"):
        return [
            "--model-path",
            str(remote["model_id"]),
            "--base-url",
            str(remote["base_url"]),
            "--model-id",
            str(remote["model_id"]),
        ]
    models = assets["models"]
    model_path = models.get("local_model")
    model_id = models.get("local_model_id")
    if not model_path:
        model_path = models.get("qwen25_vl_7b") if prefer_7b else models.get("qwen25_vl_3b")
    if not model_path:
        model_path = models.get("qwen25_vl_3b") or models.get("qwen25_vl_7b")
    cmd = [
        "--model-path",
        str(model_path),
        "--start-server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.55",
        "--max-model-len",
        "8192",
        "--server-timeout-s",
        "1800",
    ]
    if model_id:
        cmd += ["--model-id", str(model_id)]
    if server_log is not None:
        cmd += ["--server-log", str(server_log)]
    return cmd


def _require_model_or_api(assets: dict, smoke: bool = False) -> list[str]:
    remote = assets["remote_api"]
    models = assets["models"]
    if remote.get("base_url") and remote.get("model_id"):
        return []
    if models.get("local_model"):
        return []
    if models.get("qwen25_vl_3b") or models.get("qwen25_vl_7b"):
        return []
    return ["No remote OpenAI-compatible API configured and no local Qwen2.5-VL model path found."]


def _require_nextqa(assets: dict, smoke: bool = False) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    missing = []
    for key in ("video_root", "map_json", "val_csv"):
        if not nextqa.get(key):
            missing.append(f"NExT-QA {key} missing")
    val_probe = nextqa.get("val_probe") or {}
    if nextqa.get("video_root") and nextqa.get("map_json") and nextqa.get("val_csv") and not val_probe.get("ok"):
        missing.append(f"NExT-QA validation videos not resolvable ({val_probe.get('reason') or 'unknown reason'}).")
    return missing + _require_model_or_api(assets)


def _require_nextqa_captions(assets: dict, smoke: bool = False) -> list[str]:
    return _require_nextqa(assets)


def _require_videoespresso(assets: dict, smoke: bool = False) -> list[str]:
    ve = assets["datasets"]["videoespresso"]
    missing = []
    for key in ("test_json", "test_video_root"):
        if not ve.get(key):
            missing.append(f"VideoEspresso {key} missing")
    return missing + _require_model_or_api(assets)


def _require_egoschema(assets: dict, smoke: bool = False) -> list[str]:
    missing = _require_model_or_api(assets)
    if not assets["packages"].get("datasets"):
        missing.append("datasets package missing for EgoSchema HF fallback.")
    return missing


def _egoschema_local_coverage(json_path: str | None, video_root: str | None) -> dict[str, int]:
    if not json_path or not video_root:
        return {"total": 0, "present": 0, "missing": 0}
    path = Path(json_path)
    root = Path(video_root)
    if not path.exists() or not root.exists():
        return {"total": 0, "present": 0, "missing": 0}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"total": 0, "present": 0, "missing": 0}
    if not isinstance(rows, list):
        return {"total": 0, "present": 0, "missing": 0}
    total = 0
    present = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        rel = str(row.get("video_path") or row.get("video") or "").strip()
        if not rel:
            continue
        total += 1
        candidate = root / rel
        if candidate.exists() and candidate.stat().st_size > 0:
            present += 1
    return {"total": total, "present": present, "missing": max(0, total - present)}


def _use_local_egoschema(assets: dict, smoke: bool) -> bool:
    eg = assets["datasets"]["egoschema"]
    coverage = _egoschema_local_coverage(eg.get("json"), eg.get("video_root"))
    if coverage["total"] <= 0:
        return False
    if smoke:
        return coverage["present"] > 0
    return coverage["missing"] == 0


def _require_cached_video_dataset(assets: dict, dataset: str, smoke: bool = False) -> list[str]:
    missing = _require_model_or_api(assets)
    cache_root = assets["datasets"]["video_cache"].get("root")
    if not cache_root:
        return missing + [f"{dataset} video cache root missing."]
    try:
        row = _coverage(dataset, _dataset_split(dataset, ""), Path(cache_root))
    except Exception as exc:
        return missing + [f"{dataset} cache coverage check failed: {type(exc).__name__}: {exc}"]
    if row["total_videos"] == 0:
        missing.append(f"{dataset} cache coverage check found zero videos in the benchmark metadata.")
    elif smoke and row["present"] == 0:
        missing.append(
            f"{dataset} smoke run needs at least one cached video under {row['cache_dir']}; "
            f"currently present=0/{row['total_videos']}."
        )
    elif not smoke and row["missing"]:
        examples = ", ".join(row["missing_examples"][:5])
        missing.append(
            f"{dataset} full cached-only run is incomplete: "
            f"present={row['present']}/{row['total_videos']}, missing={row['missing']}. "
            f"Missing examples: {examples}"
        )
    return missing


def _require_videomme_cache(assets: dict, smoke: bool = False) -> list[str]:
    return _require_cached_video_dataset(assets, "videomme", smoke)


def _require_lvbench_cache(assets: dict, smoke: bool = False) -> list[str]:
    return _require_cached_video_dataset(assets, "lvbench", smoke)


def _require_llava_ov_cache(assets: dict, dataset: str, smoke: bool = False) -> list[str]:
    missing = _require_cached_video_dataset(assets, dataset, smoke)
    if not assets["models"].get("llava_ov_7b"):
        missing.append("Local LLaVA-OV-7B checkpoint required for Hugging Face evaluation.")
    if not assets["models"].get("llava_next_path"):
        missing.append(
            "LLaVA-NeXT source checkout required for LLaVA-OV Hugging Face evaluation "
            "(set REVISE_LLAVA_NEXT_PATH)."
        )
    return missing


def _require_llava_ov_videomme_cache(assets: dict, smoke: bool = False) -> list[str]:
    return _require_llava_ov_cache(assets, "videomme", smoke)


def _require_llava_ov_lvbench_cache(assets: dict, smoke: bool = False) -> list[str]:
    return _require_llava_ov_cache(assets, "lvbench", smoke)


def _require_nextqa_training(assets: dict, smoke: bool = False) -> list[str]:
    missing = _require_nextqa(assets)
    train_probe = assets["datasets"]["nextqa"].get("train_probe") or {}
    if not train_probe.get("ok"):
        missing.append(
            "NExT-QA train videos not resolvable; download/register the official raw train videos before SFT/GRPO."
        )
    if not assets["models"].get("qwen25_vl_3b"):
        missing.append("Local Qwen2.5-VL-3B checkpoint required for training.")
    if not assets["models"].get("qwen25_vl_7b") and not assets["models"].get("local_model") and not (
        assets["remote_api"].get("base_url") and assets["remote_api"].get("model_id")
    ):
        missing.append("Teacher generation needs a local teacher checkpoint or an OpenAI-compatible API endpoint.")
    return missing


def _require_videoespresso_training(assets: dict, smoke: bool = False) -> list[str]:
    missing = _require_videoespresso(assets)
    if not assets["datasets"]["videoespresso"].get("train_video_json"):
        missing.append("VideoEspresso open-ended train JSON missing.")
    if not assets["models"].get("qwen25_vl_3b"):
        missing.append("Local Qwen2.5-VL-3B checkpoint required for training.")
    if not assets["models"].get("qwen25_vl_7b") and not assets["models"].get("local_model") and not (
        assets["remote_api"].get("base_url") and assets["remote_api"].get("model_id")
    ):
        missing.append("Teacher generation needs a local teacher checkpoint or an OpenAI-compatible API endpoint.")
    return missing


def _cmd_nextqa_pnp_variant(
    assets: dict,
    smoke: bool,
    out_dir: Path,
    *,
    exp_id: str,
    max_rounds: int,
    max_frames_per_round: int,
    port: int,
) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/plug_and_play_nextqa_vllm.py"),
        "--video-root",
        nextqa["video_root"],
        "--map-json",
        nextqa["map_json"],
        "--csv",
        nextqa["val_csv"],
        "--seed",
        "0",
        "--max-rounds",
        "2" if smoke else str(max_rounds),
        "--max-frames-per-round",
        "3" if smoke else str(max_frames_per_round),
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / f"{exp_id}.summary.json"),
        "--log-jsonl",
        str(out_dir / f"{exp_id}.jsonl"),
    ]
    cmd += _model_endpoint_args(
        assets,
        prefer_7b=not smoke,
        port=port,
        server_log=out_dir / f"{exp_id}.server.log",
    )
    return cmd


def _cmd_nextqa_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    return _cmd_nextqa_pnp_variant(
        assets,
        smoke,
        out_dir,
        exp_id="nextqa_pnp",
        max_rounds=4,
        max_frames_per_round=3,
        port=18000,
    )


def _nextqa_pnp_variant_builder(exp_id: str, max_rounds: int, max_frames_per_round: int, port: int) -> ExperimentBuilder:
    return lambda assets, smoke, out_dir: _cmd_nextqa_pnp_variant(
        assets,
        smoke,
        out_dir,
        exp_id=exp_id,
        max_rounds=max_rounds,
        max_frames_per_round=max_frames_per_round,
        port=port,
    )


def _cmd_nextqa_oneshot(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/oneshot_local_mc_vllm.py"),
        "--dataset",
        "nextqa",
        "--video-root",
        nextqa["video_root"],
        "--map-json",
        nextqa["map_json"],
        "--csv",
        nextqa["val_csv"],
        "--max-frames",
        "8",
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "nextqa_oneshot.summary.json"),
        "--log-jsonl",
        str(out_dir / "nextqa_oneshot.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18001, server_log=out_dir / "nextqa_oneshot.server.log")
    return cmd


def _cmd_nextqa_caption(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/eval_nextqa_caption_vllm.py"),
        "--video-root",
        nextqa["video_root"],
        "--map-json",
        nextqa["map_json"],
        "--csv",
        nextqa["val_csv"],
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "nextqa_caption.summary.json"),
        "--log-jsonl",
        str(out_dir / "nextqa_caption.jsonl"),
        "--generated-captions-dir",
        str(out_dir / "generated_captions" / "nextqa"),
    ]
    if nextqa.get("captions_dir"):
        cmd += ["--captions-dir", nextqa["captions_dir"]]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=19000, server_log=out_dir / "nextqa_caption.server.log")
    return cmd


def _cmd_nextqa_videoagent(assets: dict, smoke: bool, out_dir: Path, *, official: bool) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    script = (
        "eval_nextqa_videoagent_officialstyle_caption_vllm.py"
        if official
        else "eval_nextqa_videoagent_caption_vllm.py"
    )
    prefix = "nextqa_videoagent_official" if official else "nextqa_videoagent"
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "examples/videoagent" / script),
        "--video-root",
        nextqa["video_root"],
        "--map-json",
        nextqa["map_json"],
        "--csv",
        nextqa["val_csv"],
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / f"{prefix}.summary.json"),
        "--log-jsonl",
        str(out_dir / f"{prefix}.jsonl"),
        "--generated-captions-dir",
        str(out_dir / "generated_captions" / "nextqa"),
    ]
    if nextqa.get("captions_dir"):
        cmd += ["--captions-dir", nextqa["captions_dir"]]
    cmd += _model_endpoint_args(
        assets,
        prefer_7b=False,
        port=18200 if not official else 18201,
        server_log=out_dir / f"{prefix}.server.log",
    )
    return cmd


def _cmd_videoespresso_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    return _cmd_videoespresso_pnp_variant(
        assets,
        smoke,
        out_dir,
        exp_id="videoespresso_pnp",
        max_rounds=4,
        max_frames_per_round=3,
        port=18100,
    )


def _cmd_videoespresso_pnp_variant(
    assets: dict,
    smoke: bool,
    out_dir: Path,
    *,
    exp_id: str,
    max_rounds: int,
    max_frames_per_round: int,
    port: int,
) -> list[str]:
    ve = assets["datasets"]["videoespresso"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/plug_and_play_egoschema_vllm.py"),
        "--dataset-name",
        "videoespresso",
        "--json",
        ve["test_json"],
        "--video-root",
        ve["test_video_root"],
        "--max-rounds",
        "2" if smoke else str(max_rounds),
        "--max-frames-per-round",
        "3" if smoke else str(max_frames_per_round),
        "--max-samples",
        "1" if smoke else "0",
        "--videoespresso-use-official-prompt",
        "--no-videoespresso-with-evidence",
        "--summary-json",
        str(out_dir / f"{exp_id}.summary.json"),
        "--log-jsonl",
        str(out_dir / f"{exp_id}.jsonl"),
    ]
    cmd += _model_endpoint_args(
        assets,
        prefer_7b=not smoke,
        port=port,
        server_log=out_dir / f"{exp_id}.server.log",
    )
    return cmd


def _videoespresso_pnp_variant_builder(
    exp_id: str,
    max_rounds: int,
    max_frames_per_round: int,
    port: int,
) -> ExperimentBuilder:
    return lambda assets, smoke, out_dir: _cmd_videoespresso_pnp_variant(
        assets,
        smoke,
        out_dir,
        exp_id=exp_id,
        max_rounds=max_rounds,
        max_frames_per_round=max_frames_per_round,
        port=port,
    )


# Component ablation (overleaf/tables/abaltion_components.tex) holds the budget at the
# REVISE default of 4 rounds x 3 frames and toggles state carryover / structured summary.
_COMPONENT_ABLATION_ROUNDS = 4
_COMPONENT_ABLATION_FRAMES = 3


def _cmd_videoespresso_components_variant(
    assets: dict,
    smoke: bool,
    out_dir: Path,
    *,
    exp_id: str,
    ablate_carryover: bool,
    ablate_structured: bool,
    port: int,
) -> list[str]:
    ve = assets["datasets"]["videoespresso"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/plug_and_play_egoschema_vllm.py"),
        "--dataset-name",
        "videoespresso",
        "--json",
        ve["test_json"],
        "--video-root",
        ve["test_video_root"],
        "--max-rounds",
        "2" if smoke else str(_COMPONENT_ABLATION_ROUNDS),
        "--max-frames-per-round",
        "3" if smoke else str(_COMPONENT_ABLATION_FRAMES),
        "--max-samples",
        "1" if smoke else "0",
        "--videoespresso-use-official-prompt",
        "--no-videoespresso-with-evidence",
    ]
    if ablate_carryover:
        cmd.append("--ablate-state-carryover")
    if ablate_structured:
        cmd.append("--ablate-structured-summary")
    cmd += [
        "--summary-json",
        str(out_dir / f"{exp_id}.summary.json"),
        "--log-jsonl",
        str(out_dir / f"{exp_id}.jsonl"),
    ]
    cmd += _model_endpoint_args(
        assets,
        prefer_7b=not smoke,
        port=port,
        server_log=out_dir / f"{exp_id}.server.log",
    )
    return cmd


def _videoespresso_components_variant_builder(
    exp_id: str,
    ablate_carryover: bool,
    ablate_structured: bool,
    port: int,
) -> ExperimentBuilder:
    return lambda assets, smoke, out_dir: _cmd_videoespresso_components_variant(
        assets,
        smoke,
        out_dir,
        exp_id=exp_id,
        ablate_carryover=ablate_carryover,
        ablate_structured=ablate_structured,
        port=port,
    )


def _cmd_videoespresso_oneshot(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    ve = assets["datasets"]["videoespresso"]
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/oneshot_local_mc_vllm.py"),
        "--dataset",
        "jsonmc",
        "--dataset-name",
        "videoespresso",
        "--json",
        ve["test_json"],
        "--video-root",
        ve["test_video_root"],
        "--max-frames",
        "8",
        "--max-samples",
        "1" if smoke else "0",
        "--videoespresso-use-official-prompt",
        "--no-videoespresso-with-evidence",
        "--summary-json",
        str(out_dir / "videoespresso_oneshot.summary.json"),
        "--log-jsonl",
        str(out_dir / "videoespresso_oneshot.jsonl"),
    ]
    cmd += _model_endpoint_args(
        assets,
        prefer_7b=False,
        port=18101,
        server_log=out_dir / "videoespresso_oneshot.server.log",
    )
    return cmd


def _cmd_egoschema_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    eg = assets["datasets"]["egoschema"]
    cache_dir = str(Path(assets["asset_root"]) / "EgoSchema" / "videos")
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/plug_and_play_egoschema_vllm.py"),
        "--dataset-name",
        "egoschema",
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--max-samples",
        "1" if smoke else "0",
        "--summary-json",
        str(out_dir / "egoschema_pnp.summary.json"),
        "--log-jsonl",
        str(out_dir / "egoschema_pnp.jsonl"),
    ]
    if _use_local_egoschema(assets, smoke):
        cmd += [
            "--egoschema-source",
            "local",
            "--json",
            eg["json"],
            "--video-root",
            eg["video_root"],
        ]
    else:
        cmd += [
            "--egoschema-source",
            "hf",
            "--egoschema-hf-config",
            "Subset",
            "--egoschema-video-cache-dir",
            cache_dir,
            "--auto-download-egoschema-videos",
        ]
    cmd += _model_endpoint_args(assets, prefer_7b=not smoke, port=18110, server_log=out_dir / "egoschema_pnp.server.log")
    return cmd


def _cmd_videomme_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/plug_and_play_videomme_lvbench_vllm.py"),
        "--dataset",
        "videomme",
        "--cached-only",
        "--max-samples",
        "1" if smoke else "0",
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or str(REPO_ROOT / "data/revise_assets/video_cache"),
        "--videomme-use-official-prompt",
        "--summary-json",
        str(out_dir / "videomme_pnp.summary.json"),
        "--log-jsonl",
        str(out_dir / "videomme_pnp.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18120, server_log=out_dir / "videomme_pnp.server.log")
    return cmd


def _cmd_lvbench_pnp(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/plug_and_play_videomme_lvbench_vllm.py"),
        "--dataset",
        "lvbench",
        "--cached-only",
        "--max-samples",
        "1" if smoke else "0",
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or str(REPO_ROOT / "data/revise_assets/video_cache"),
        "--summary-json",
        str(out_dir / "lvbench_pnp.summary.json"),
        "--log-jsonl",
        str(out_dir / "lvbench_pnp.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18121, server_log=out_dir / "lvbench_pnp.server.log")
    return cmd


def _cmd_videomme_pnp_hf(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/plug_and_play_lvbench_hf.py"),
        "--dataset",
        "videomme",
        "--max-samples",
        "1" if smoke else "0",
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or str(REPO_ROOT / "data/revise_assets/video_cache"),
        "--videomme-use-official-prompt",
        "--model-path",
        assets["models"]["llava_ov_7b"],
        "--dtype",
        "bfloat16",
        "--summary-json",
        str(out_dir / "videomme_pnp_hf.summary.json"),
        "--log-jsonl",
        str(out_dir / "videomme_pnp_hf.jsonl"),
    ]
    return cmd


def _cmd_lvbench_pnp_hf(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/plug_and_play_lvbench_hf.py"),
        "--dataset",
        "lvbench",
        "--max-samples",
        "1" if smoke else "0",
        "--max-rounds",
        "2" if smoke else "4",
        "--max-frames-per-round",
        "3",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or str(REPO_ROOT / "data/revise_assets/video_cache"),
        "--model-path",
        assets["models"]["llava_ov_7b"],
        "--dtype",
        "bfloat16",
        "--summary-json",
        str(out_dir / "lvbench_pnp_hf.summary.json"),
        "--log-jsonl",
        str(out_dir / "lvbench_pnp_hf.jsonl"),
    ]
    return cmd


def _cmd_videomme_oneshot(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/oneshot_videomme_lvbench_vllm.py"),
        "--dataset",
        "videomme",
        "--cached-only",
        "--max-samples",
        "1" if smoke else "0",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or str(REPO_ROOT / "data/revise_assets/video_cache"),
        "--videomme-use-official-prompt",
        "--summary-json",
        str(out_dir / "videomme_oneshot.summary.json"),
        "--log-jsonl",
        str(out_dir / "videomme_oneshot.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18122, server_log=out_dir / "videomme_oneshot.server.log")
    return cmd


def _cmd_lvbench_oneshot(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "revise/oneshot_videomme_lvbench_vllm.py"),
        "--dataset",
        "lvbench",
        "--cached-only",
        "--max-samples",
        "1" if smoke else "0",
        "--video-cache-dir",
        assets["datasets"]["video_cache"]["root"] or str(REPO_ROOT / "data/revise_assets/video_cache"),
        "--summary-json",
        str(out_dir / "lvbench_oneshot.summary.json"),
        "--log-jsonl",
        str(out_dir / "lvbench_oneshot.jsonl"),
    ]
    cmd += _model_endpoint_args(assets, prefer_7b=False, port=18123, server_log=out_dir / "lvbench_oneshot.server.log")
    return cmd


def _cmd_nextqa_hydra_resolve(assets: dict, smoke: bool, out_dir: Path) -> list[str]:
    nextqa = assets["datasets"]["nextqa"]
    model_path = assets["models"]["qwen25_vl_3b"]
    return [
        _python_bin(),
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        str(REPO_ROOT / "revise/config"),
        "--config-name",
        "revise_nextqa_smoke",
        f"data.nextqa.video_root={nextqa['video_root']}",
        f"data.nextqa.map_json={nextqa['map_json']}",
        f"data.train_files={nextqa['train_csv']}",
        f"data.val_files={nextqa['val_csv']}",
        f"actor_rollout_ref.model.path={model_path}",
        "--cfg",
        "job",
        "--resolve",
    ]


def _shell_env(**items: str) -> str:
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in items.items() if value is not None)


def _teacher_env(assets: dict) -> dict[str, Optional[str]]:
    remote = assets["remote_api"]
    if remote.get("base_url") and remote.get("model_id"):
        model_id = str(remote["model_id"])
        return {
            "TEACHER_BASE_URL": str(remote["base_url"]),
            "TEACHER_MODEL_ID": model_id,
            "TEACHER_MODEL_PATH": model_id,
        }
    models = assets["models"]
    model_path = models.get("local_model") or models.get("qwen25_vl_7b") or models.get("qwen25_vl_3b")
    return {
        "TEACHER_MODEL_PATH": str(model_path) if model_path else None,
        "TEACHER_MODEL_ID": str(models.get("local_model_id")) if models.get("local_model_id") else None,
    }


def _external_teacher_step(env_var: str, teacher_log: Path) -> str | None:
    """If `env_var` is set in the render-time environment, return a shell
    snippet that validates the path exists and symlinks it into the place
    the SFT step expects, replacing the inline teacher-generation step.

    Use case (Phase A -> Phase E handoff): Phase A produces a high-quality
    teacher JSONL via a separately-submitted Slurm job (e.g. Qwen2.5-VL-72B
    AWQ). Phase E should consume that JSONL instead of re-running teacher
    generation inline at a smaller model size. Setting REVISE_*_TEACHER_LOG_OVERRIDE
    when invoking paper_suite.py (or submit_paper_suite_slurm.py) plumbs the
    Phase A artifact in.

    Returns the substitute shell command, or None if the env var is unset.
    """
    override = os.environ.get(env_var)
    if not override:
        return None
    override_path = Path(override)
    parent = teacher_log.parent
    # Fail loudly at sbatch-run time if the override path is missing — we
    # would otherwise silently train on no data. ln -sfn (no-dereference)
    # so a re-render after Phase A reruns updates the link.
    return (
        f"mkdir -p {shlex.quote(str(parent))} && "
        f"if [ ! -f {shlex.quote(str(override_path))} ]; then "
        f'echo "ERROR: Phase A teacher JSONL not found: {shlex.quote(str(override_path))}" >&2; exit 1; '
        f"fi && "
        f"ln -sfn {shlex.quote(str(override_path))} {shlex.quote(str(teacher_log))}"
    )


def _manual_nextqa_pipeline(assets: dict, smoke: bool, out_dir: Path) -> str:
    nextqa = assets["datasets"]["nextqa"]
    python_bin = _python_bin()
    teacher_log = out_dir / "teacher_logs" / ("nextqa_teacher_smoke.jsonl" if smoke else "nextqa_teacher.jsonl")
    teacher_server_log = out_dir / "server_logs" / (
        "nextqa_teacher_smoke.server.log" if smoke else "nextqa_teacher.server.log"
    )
    sft_output = out_dir / "sft_data" / "nextqa_revise_sft.parquet"
    sft_dir = out_dir / "checkpoints" / "nextqa_sft"
    rl_dir = out_dir / "checkpoints" / "nextqa_grpo_after_sft"
    max_samples = "4" if smoke else "8000"
    n_gpus = "1" if smoke else "4"
    sft_extra = [
        f"data.train_files={sft_output.parent / 'nextqa_revise_sft_train.parquet'}",
        f"data.val_files={sft_output.parent / 'nextqa_revise_sft_val.parquet'}",
        f"trainer.default_local_dir={sft_dir}",
    ]
    rl_extra = [f"trainer.default_local_dir={rl_dir}"]
    rl_extra.append("actor_rollout_ref.rollout.agent.num_workers=1")
    if smoke:
        sft_extra += [
            "data.train_batch_size=1",
            "data.micro_batch_size_per_gpu=1",
            "trainer.total_epochs=1",
            "trainer.total_training_steps=1",
            "trainer.n_gpus_per_node=1",
            "trainer.test_freq=-1",
            "trainer.logger=[console]",
        ]
        rl_extra += [
            "data.train_batch_size=1",
            "data.max_prompt_length=4096",
            "data.max_response_length=512",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.data_parallel_size=1",
            "actor_rollout_ref.rollout.n=1",
            "actor_rollout_ref.actor.ppo_mini_batch_size=1",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            "trainer.total_training_steps=1",
            "trainer.n_gpus_per_node=1",
            "trainer.logger=[console]",
            "ray_kwargs.ray_init.num_cpus=8",
        ]
    external_teacher = _external_teacher_step("REVISE_NEXTQA_TEACHER_LOG_OVERRIDE", teacher_log)
    teacher_step = external_teacher if external_teacher else (
        _shell_env(
            PYTHON_BIN=str(python_bin),
            VIDEO_ROOT=str(nextqa["video_root"]),
            MAP_JSON=str(nextqa["map_json"]),
            CSV=str(nextqa["train_csv"]),
            MAX_SAMPLES=max_samples,
            LOG_PATH=str(teacher_log),
            SERVER_LOG=str(teacher_server_log),
            SERVER_TIMEOUT_S="1800",
            TENSOR_PARALLEL_SIZE="1",
            GPU_MEMORY_UTILIZATION="0.55",
            **_teacher_env(assets),
        )
        + " ./revise/run_generate_teacher_data.sh"
    )
    return (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        + teacher_step
        + " && "
        + _shell_env(
            PYTHON_BIN=str(python_bin),
            TEACHER_LOG=str(teacher_log),
            SFT_INPUT=str(teacher_log),
            SFT_OUTPUT=str(sft_output),
            SFT_CKPT_DIR=str(sft_dir),
            N_GPUS=n_gpus,
            ENGINE="vllm",
            SFT_EXTRA_ARGS=" ".join(sft_extra),
            RL_EXTRA_ARGS=" ".join(rl_extra),
        )
        + " "
        "./revise/run_revise_nextqa_sft_then_rl.sh "
        "data.nextqa.video_root="
        + shlex.quote(str(nextqa["video_root"]))
        + " "
        "data.nextqa.map_json="
        + shlex.quote(str(nextqa["map_json"]))
        + " "
        "data.train_files="
        + shlex.quote(str(nextqa["train_csv"]))
        + " "
        "data.val_files="
        + shlex.quote(str(nextqa["val_csv"]))
    )


def _manual_videoespresso_pipeline(assets: dict, smoke: bool, out_dir: Path) -> str:
    ve = assets["datasets"]["videoespresso"]
    python_bin = _python_bin()
    mc_json = str(out_dir / "prepared" / ("videoespresso_train_mc_smoke.json" if smoke else "videoespresso_train_mc.json"))
    teacher_log = out_dir / "teacher_logs" / (
        "videoespresso_teacher_smoke.jsonl" if smoke else "videoespresso_teacher.jsonl"
    )
    teacher_server_log = out_dir / "server_logs" / (
        "videoespresso_teacher_smoke.server.log" if smoke else "videoespresso_teacher.server.log"
    )
    sft_output = out_dir / "sft_data" / "videoespresso_revise_sft.parquet"
    sft_dir = out_dir / "checkpoints" / "videoespresso_sft"
    rl_dir = out_dir / "checkpoints" / "videoespresso_grpo_after_sft"
    train_video_root = Path(str(ve["root"])) / "train_video"
    max_samples = "4" if smoke else "8000"
    n_gpus = "1" if smoke else "4"
    prepare_cap = " --max-rows 16" if smoke else ""
    sft_extra = [
        f"data.train_files={sft_output.parent / 'videoespresso_revise_sft_train.parquet'}",
        f"data.val_files={sft_output.parent / 'videoespresso_revise_sft_val.parquet'}",
        f"trainer.default_local_dir={sft_dir}",
    ]
    rl_extra = [f"trainer.default_local_dir={rl_dir}"]
    rl_extra.append("actor_rollout_ref.rollout.agent.num_workers=1")
    if smoke:
        sft_extra += [
            "data.train_batch_size=1",
            "data.micro_batch_size_per_gpu=1",
            "trainer.total_epochs=1",
            "trainer.total_training_steps=1",
            "trainer.n_gpus_per_node=1",
            "trainer.test_freq=-1",
            "trainer.logger=[console]",
        ]
        rl_extra += [
            "data.train_batch_size=1",
            "data.max_prompt_length=4096",
            "data.max_response_length=512",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.data_parallel_size=1",
            "actor_rollout_ref.rollout.n=1",
            "actor_rollout_ref.actor.ppo_mini_batch_size=1",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            "trainer.total_training_steps=1",
            "trainer.n_gpus_per_node=1",
            "trainer.logger=[console]",
            "ray_kwargs.ray_init.num_cpus=8",
        ]
    return (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        + shlex.quote(str(python_bin))
        + " "
        + shlex.quote(str(REPO_ROOT / "scripts/prepare_videoespresso_mc_train.py"))
        + " "
        f"--input {shlex.quote(str(ve['train_video_json']))} "
        f"--output {shlex.quote(str(mc_json))}"
        + prepare_cap
        + " && "
        + shlex.quote(str(python_bin))
        + " "
        + shlex.quote(str(REPO_ROOT / "scripts/extract_videoespresso_split_zip_subset.py"))
        + " "
        f"--json {shlex.quote(str(mc_json))} "
        f"--archive-dir {shlex.quote(str(train_video_root))} "
        f"--out-root {shlex.quote(str(train_video_root / 'all_video'))} "
        f"--manifest {shlex.quote(str(out_dir / 'prepared' / 'videoespresso_train_video_extract_manifest.json'))}"
        + " && "
        + (
            _external_teacher_step("REVISE_VIDEOESPRESSO_TEACHER_LOG_OVERRIDE", teacher_log)
            or (
                _shell_env(
                    PYTHON_BIN=str(python_bin),
                    VIDEO_ROOT=str(ve["root"]),
                    JSON=str(mc_json),
                    MAX_SAMPLES=max_samples,
                    LOG_PATH=str(teacher_log),
                    SERVER_LOG=str(teacher_server_log),
                    SERVER_TIMEOUT_S="1800",
                    TENSOR_PARALLEL_SIZE="1",
                    GPU_MEMORY_UTILIZATION="0.55",
                    **_teacher_env(assets),
                )
                + " ./revise/run_generate_teacher_data_videoespresso.sh"
            )
        )
        + " && "
        + _shell_env(
            PYTHON_BIN=str(python_bin),
            TEACHER_LOG=str(teacher_log),
            SFT_INPUT=str(teacher_log),
            SFT_OUTPUT=str(sft_output),
            SFT_CKPT_DIR=str(sft_dir),
            N_GPUS=n_gpus,
            ENGINE="vllm",
            SFT_EXTRA_ARGS=" ".join(sft_extra),
            RL_EXTRA_ARGS=" ".join(rl_extra),
        )
        + " "
        "./revise/run_revise_videoespresso_sft_then_rl.sh "
        "data.local_mc.video_root="
        + shlex.quote(str(ve["root"]))
        + " "
        "data.train_files="
        + shlex.quote(str(mc_json))
        + " "
        "data.val_files="
        + shlex.quote(str(ve["test_json"]))
    )


def _require_lvbench_training_cache(assets: dict, smoke: bool = False) -> list[str]:
    # GRPO training always reads the full LVBench corpus; the smoke shortcut
    # (one cached video) is for PNP eval only. Ignore the smoke flag so
    # `paper_suite.py check --smoke` does not falsely report ok when only
    # a partial cache is present.
    return _require_lvbench_cache(assets, smoke=False)


def _lvbench_reward_ablation_builder(config_name: str, exp_label: str):
    """Returns an ExperimentBuilder that runs GRPO training for one LVBench reward ablation.

    config_name: hydra --config-name (e.g. "revise_lvbench_grpo_eager_paper_l1_s1_stop0p5_beta1_tau2_100")
    exp_label:   short label used in the per-run output directory
    """

    def _build(assets: dict, smoke: bool, out_dir: Path) -> str:
        python_bin = _python_bin()
        config_dir = REPO_ROOT / "revise/config"
        ckpt_dir = out_dir / "checkpoints" / f"lvbench_grpo_{exp_label}"
        video_cache_root = (
            assets["datasets"]["video_cache"].get("root")
            or str(REPO_ROOT / "data/revise_assets/video_cache")
        )
        env_str = _shell_env(
            REVISE_ASSET_ROOT=str(assets.get("asset_root", "")),
            REVISE_VIDEO_CACHE_DIR=str(video_cache_root),
            PYTHONUNBUFFERED="1",
            TOKENIZERS_PARALLELISM="false",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
        ).strip()
        train_cmd = (
            f"{shlex.quote(str(python_bin))} -m verl.trainer.main_ppo "
            f"--config-path {shlex.quote(str(config_dir))} "
            f"--config-name {config_name} "
            f"trainer.default_local_dir={shlex.quote(str(ckpt_dir))}"
        )
        # Note: smoke-run path is intentionally omitted. Reward-ablation
        # GRPO is run_supported=False — it must be launched through Slurm,
        # not subprocess.run from paper_suite. A partial smoke override
        # here would mislead readers into thinking `--smoke` produces a
        # cheap dry-run, when it would still request the full 4-GPU
        # tensor-parallel allocation declared in the YAML.
        _ = smoke  # silence unused-arg lint without changing the signature
        # Env vars are prefixed inline to the python invocation (single command)
        # so they actually apply to the trainer; do not separate with && or they
        # become a no-op statement. Matches the pattern used by _manual_*_pipeline.
        prefixed = f"{env_str} {train_cmd}" if env_str else train_cmd
        return f"cd {shlex.quote(str(REPO_ROOT))} && {prefixed}"

    return _build


EXPERIMENTS: dict[str, dict[str, object]] = {
    "nextqa_pnp": {
        "title": "NExT-QA plug-and-play",
        "paper_ref": "paper tables: NExT-QA, RL Results",
        "check": _require_nextqa,
        "build": _cmd_nextqa_pnp,
        "run_supported": True,
    },
    "nextqa_oneshot": {
        "title": "NExT-QA direct reasoning (one-shot)",
        "paper_ref": "paper table: RL Results",
        "check": _require_nextqa,
        "build": _cmd_nextqa_oneshot,
        "run_supported": True,
    },
    "nextqa_caption": {
        "title": "NExT-QA caption-only baseline",
        "paper_ref": "paper table: caption ablation",
        "check": _require_nextqa_captions,
        "build": _cmd_nextqa_caption,
        "run_supported": True,
    },
    "nextqa_videoagent": {
        "title": "NExT-QA VideoAgent-style caption baseline",
        "paper_ref": "paper table: caption ablation",
        "check": _require_nextqa_captions,
        "build": lambda assets, smoke, out_dir: _cmd_nextqa_videoagent(assets, smoke, out_dir, official=False),
        "run_supported": True,
    },
    "nextqa_videoagent_official": {
        "title": "NExT-QA VideoAgent official-style baseline",
        "paper_ref": "paper table: VideoAgent baseline",
        "check": _require_nextqa_captions,
        "build": lambda assets, smoke, out_dir: _cmd_nextqa_videoagent(assets, smoke, out_dir, official=True),
        "run_supported": True,
    },
    "videoespresso_pnp": {
        "title": "VideoEspresso plug-and-play",
        "paper_ref": "paper tables: VideoEspresso, RL Results",
        "check": _require_videoespresso,
        "build": _cmd_videoespresso_pnp,
        "run_supported": True,
    },
    "videoespresso_budget_10x2": {
        "title": "VideoEspresso budget ablation 10 rounds x 2 frames",
        "paper_ref": "paper table: turn/frame budget ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_budget_10x2", 10, 2, 18300),
        "run_supported": True,
    },
    "videoespresso_budget_2x10": {
        "title": "VideoEspresso budget ablation 2 rounds x 10 frames",
        "paper_ref": "paper table: turn/frame budget ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_budget_2x10", 2, 10, 18301),
        "run_supported": True,
    },
    "videoespresso_budget_4x5": {
        "title": "VideoEspresso budget ablation 4 rounds x 5 frames",
        "paper_ref": "paper table: turn/frame budget ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_budget_4x5", 4, 5, 18302),
        "run_supported": True,
    },
    "videoespresso_budget_5x4": {
        "title": "VideoEspresso budget ablation 5 rounds x 4 frames",
        "paper_ref": "paper table: turn/frame budget ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_budget_5x4", 5, 4, 18303),
        "run_supported": True,
    },
    "videoespresso_budget_7x4": {
        "title": "VideoEspresso budget ablation 7 rounds x 4 frames",
        "paper_ref": "paper table: turn/frame budget ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_budget_7x4", 7, 4, 18304),
        "run_supported": True,
    },
    "videoespresso_budget_9x4": {
        "title": "VideoEspresso budget ablation 9 rounds x 4 frames",
        "paper_ref": "paper table: turn/frame budget ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_budget_9x4", 9, 4, 18305),
        "run_supported": True,
    },
    "videoespresso_best_01x06": {
        "title": "VideoEspresso best-config sweep candidate 1 round x 6 frames",
        "paper_ref": "paper table: best frames/rounds ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_best_01x06", 1, 6, 18310),
        "run_supported": True,
    },
    "videoespresso_best_02x04": {
        "title": "VideoEspresso best-config sweep candidate 2 rounds x 4 frames",
        "paper_ref": "paper table: best frames/rounds ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_best_02x04", 2, 4, 18311),
        "run_supported": True,
    },
    "videoespresso_best_03x06": {
        "title": "VideoEspresso best-config sweep candidate 3 rounds x 6 frames",
        "paper_ref": "paper table: best frames/rounds ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_best_03x06", 3, 6, 18312),
        "run_supported": True,
    },
    "videoespresso_best_04x04": {
        "title": "VideoEspresso best-config sweep candidate 4 rounds x 4 frames",
        "paper_ref": "paper table: best frames/rounds ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_pnp_variant_builder("videoespresso_best_04x04", 4, 4, 18313),
        "run_supported": True,
    },
    "videoespresso_components_full": {
        "title": "VideoEspresso component ablation: full structured summary-as-state",
        "paper_ref": "paper table: component ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_components_variant_builder(
            "videoespresso_components_full", ablate_carryover=False, ablate_structured=False, port=18320
        ),
        "run_supported": True,
    },
    "videoespresso_components_no_carryover": {
        "title": "VideoEspresso component ablation: no state carryover",
        "paper_ref": "paper table: component ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_components_variant_builder(
            "videoespresso_components_no_carryover", ablate_carryover=True, ablate_structured=False, port=18321
        ),
        "run_supported": True,
    },
    "videoespresso_components_no_structured": {
        "title": "VideoEspresso component ablation: no structured P/O/H/U/R fields",
        "paper_ref": "paper table: component ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_components_variant_builder(
            "videoespresso_components_no_structured", ablate_carryover=False, ablate_structured=True, port=18322
        ),
        "run_supported": True,
    },
    "videoespresso_components_no_both": {
        "title": "VideoEspresso component ablation: no state carryover and no structured fields",
        "paper_ref": "paper table: component ablation",
        "check": _require_videoespresso,
        "build": _videoespresso_components_variant_builder(
            "videoespresso_components_no_both", ablate_carryover=True, ablate_structured=True, port=18323
        ),
        "run_supported": True,
    },
    "videoespresso_oneshot": {
        "title": "VideoEspresso direct reasoning (one-shot)",
        "paper_ref": "paper table: RL Results",
        "check": _require_videoespresso,
        "build": _cmd_videoespresso_oneshot,
        "run_supported": True,
    },
    "egoschema_pnp": {
        "title": "EgoSchema plug-and-play",
        "paper_ref": "paper table: EgoSchema",
        "check": _require_egoschema,
        "build": _cmd_egoschema_pnp,
        "run_supported": True,
    },
    "videomme_pnp": {
        "title": "Video-MME plug-and-play",
        "paper_ref": "paper table: additional benchmarks",
        "check": _require_videomme_cache,
        "build": _cmd_videomme_pnp,
        "run_supported": True,
    },
    "lvbench_pnp": {
        "title": "LVBench plug-and-play",
        "paper_ref": "paper table: additional benchmarks",
        "check": _require_lvbench_cache,
        "build": _cmd_lvbench_pnp,
        "run_supported": True,
    },
    "videomme_pnp_hf": {
        "title": "Video-MME plug-and-play Hugging Face local model",
        "paper_ref": "paper table: additional benchmarks / LLaVA-OV-7B",
        "check": _require_llava_ov_videomme_cache,
        "build": _cmd_videomme_pnp_hf,
        "run_supported": True,
    },
    "lvbench_pnp_hf": {
        "title": "LVBench plug-and-play Hugging Face local model",
        "paper_ref": "paper table: additional benchmarks / LLaVA-OV-7B",
        "check": _require_llava_ov_lvbench_cache,
        "build": _cmd_lvbench_pnp_hf,
        "run_supported": True,
    },
    "videomme_oneshot": {
        "title": "Video-MME one-shot baseline",
        "paper_ref": "paper table: additional benchmarks",
        "check": _require_videomme_cache,
        "build": _cmd_videomme_oneshot,
        "run_supported": True,
    },
    "lvbench_oneshot": {
        "title": "LVBench one-shot baseline",
        "paper_ref": "paper table: additional benchmarks",
        "check": _require_lvbench_cache,
        "build": _cmd_lvbench_oneshot,
        "run_supported": True,
    },
    "nextqa_hydra_resolve": {
        "title": "NExT-QA Hydra config resolve",
        "paper_ref": "paper Sec. 4 experiments",
        "check": _require_nextqa_training,
        "build": _cmd_nextqa_hydra_resolve,
        "run_supported": True,
    },
    "nextqa_train_pipeline": {
        "title": "NExT-QA SFT+GRPO training pipeline",
        "paper_ref": "paper table: RL Results",
        "check": _require_nextqa_training,
        "build": _manual_nextqa_pipeline,
        "run_supported": False,
    },
    "videoespresso_train_pipeline": {
        "title": "VideoEspresso SFT+GRPO training pipeline",
        "paper_ref": "paper table: RL Results",
        "check": _require_videoespresso_training,
        "build": _manual_videoespresso_pipeline,
        "run_supported": False,
    },
    "lvbench_reward_ablation_row0_base_no_rl": {
        # The audit doc describes this row as "SFT-only checkpoint eval",
        # but the LVBench codepath starts GRPO directly from the HF model
        # (no SFT step), so the "no RL" baseline IS the pretrained model
        # via plug-and-play eval. This entry aliases lvbench_pnp_hf for
        # that purpose.
        "title": "LVBench reward ablation row 0: base (no RL) - pretrained LLaVA-OV eval (no SFT/no GRPO)",
        "paper_ref": "overleaf/tables/abaltion_reward.tex row: Base (no RL)",
        "check": _require_llava_ov_lvbench_cache,
        "build": _cmd_lvbench_pnp_hf,
        "run_supported": True,
    },
    "lvbench_reward_ablation_row1_full_beta1_tau2": {
        "title": "LVBench reward ablation row 1: EAGER full (l1=1, l2=1, l3=0.5, beta=1, tau=2)",
        "paper_ref": "overleaf/tables/abaltion_reward.tex row: EAGER (1,1,0.5)",
        "check": _require_lvbench_training_cache,
        "build": _lvbench_reward_ablation_builder(
            "revise_lvbench_grpo_eager_paper_l1_s1_stop0p5_beta1_tau2_100",
            "row1_l1_s1_stop0p5_b1_t2",
        ),
        "run_supported": False,
    },
    "lvbench_reward_ablation_row2_no_conf_beta1_tau2": {
        "title": "LVBench reward ablation row 2: no confidence (l1=0, l2=1, l3=0.5, beta=1, tau=2)",
        "paper_ref": "overleaf/tables/abaltion_reward.tex row: EAGER (0,1,0.5)",
        "check": _require_lvbench_training_cache,
        "build": _lvbench_reward_ablation_builder(
            "revise_lvbench_grpo_eager_paper_l0_s1_stop0p5_beta1_tau2_100",
            "row2_l0_s1_stop0p5_b1_t2",
        ),
        "run_supported": False,
    },
    "lvbench_reward_ablation_row3_no_sum_beta1_tau2": {
        "title": "LVBench reward ablation row 3: no summary (l1=1, l2=0, l3=0.5, beta=1, tau=2)",
        "paper_ref": "overleaf/tables/abaltion_reward.tex row: EAGER (1,0,0.5)",
        "check": _require_lvbench_training_cache,
        "build": _lvbench_reward_ablation_builder(
            "revise_lvbench_grpo_eager_paper_l1_s0_stop0p5_beta1_tau2_100",
            "row3_l1_s0_stop0p5_b1_t2",
        ),
        "run_supported": False,
    },
    "lvbench_reward_ablation_row4_no_stop_beta1_tau2": {
        "title": "LVBench reward ablation row 4: no early-stop (l1=1, l2=1, l3=0, beta=1, tau=2)",
        "paper_ref": "overleaf/tables/abaltion_reward.tex row: EAGER (1,1,0)",
        "check": _require_lvbench_training_cache,
        "build": _lvbench_reward_ablation_builder(
            "revise_lvbench_grpo_eager_paper_l1_s1_stop0_beta1_tau2_100",
            "row4_l1_s1_stop0_b1_t2",
        ),
        "run_supported": False,
    },
    "lvbench_reward_ablation_row5_stop_design_beta0_tau1": {
        "title": "LVBench reward ablation row 5: stop design beta=0, tau=1 (l1=0, l2=1, l3=0.5)",
        "paper_ref": "overleaf/tables/abaltion_reward.tex row: beta=0, tau=1",
        "check": _require_lvbench_training_cache,
        "build": _lvbench_reward_ablation_builder(
            "revise_lvbench_grpo_eager_paper_l0_s1_stop0p5_beta0_tau1_100",
            "row5_l0_s1_stop0p5_b0_t1",
        ),
        "run_supported": False,
    },
    "lvbench_reward_ablation_row6_stop_design_beta0_tau3": {
        "title": "LVBench reward ablation row 6: stop design beta=0, tau=3 (l1=0, l2=1, l3=0.5)",
        "paper_ref": "overleaf/tables/abaltion_reward.tex row: beta=0, tau=3",
        "check": _require_lvbench_training_cache,
        "build": _lvbench_reward_ablation_builder(
            "revise_lvbench_grpo_eager_paper_l0_s1_stop0p5_beta0_tau3_100",
            "row6_l0_s1_stop0p5_b0_t3",
        ),
        "run_supported": False,
    },
    "lvbench_reward_ablation_row7_stop_design_beta1_tau2": {
        # Row 7 has identical reward_kwargs to row 2 (lambda_conf=0,
        # lambda_sum=1, lambda_stop=0.5, beta=1, tau=2). The paper lists it
        # twice for stop-design table readability; the training run is the
        # SAME. exp_label is intentionally collapsed onto row 2 so both
        # entries target the same checkpoint dir — accidental double-submit
        # then collides cleanly on the path instead of silently spending
        # another ~4-GPU * 24h training the same model.
        "title": "LVBench reward ablation row 7: stop design beta=1, tau=2 - alias of row 2 (shared training run; do not submit independently)",
        "paper_ref": "overleaf/tables/abaltion_reward.tex row: beta=1, tau=2",
        "check": _require_lvbench_training_cache,
        "build": _lvbench_reward_ablation_builder(
            "revise_lvbench_grpo_eager_paper_l0_s1_stop0p5_beta1_tau2_100",
            "row2_l0_s1_stop0p5_b1_t2",
        ),
        "run_supported": False,
    },
}


def _selected_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(EXPERIMENTS.keys())
    if not args.experiment:
        raise SystemExit("Specify --experiment or --all.")
    unknown = [eid for eid in args.experiment if eid not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment ids: {', '.join(unknown)}")
    return list(args.experiment)


def cmd_to_text(cmd: list[str] | str) -> str:
    if isinstance(cmd, str):
        return cmd
    return shlex.join(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="List/check/run paper experiments.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    check_ap = sub.add_parser("check")
    check_ap.add_argument("--experiment", action="append")
    check_ap.add_argument("--all", action="store_true")
    check_ap.add_argument("--smoke", action="store_true", help="Apply smoke-run data availability rules.")

    run_ap = sub.add_parser("run")
    run_ap.add_argument("--experiment", action="append")
    run_ap.add_argument("--all", action="store_true")
    run_ap.add_argument("--smoke", action="store_true", help="Run the smallest supported variant.")
    run_ap.add_argument("--dry-run", action="store_true")
    run_ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "paper_suite"))

    args = ap.parse_args()
    assets = discover_assets()

    if args.cmd == "list":
        for exp_id, meta in EXPERIMENTS.items():
            print(f"{exp_id}\t{meta['title']}\t{meta['paper_ref']}")
        return 0

    if args.cmd == "check":
        for exp_id in _selected_ids(args):
            meta = EXPERIMENTS[exp_id]
            missing = meta["check"](assets, bool(args.smoke))  # type: ignore[index]
            status = "ok" if not missing else "blocked"
            print(f"[{status}] {exp_id}: {meta['title']}")
            if missing:
                for item in missing:
                    print(f"  - {item}")
            print(f"  command: {cmd_to_text(meta['build'](assets, bool(args.smoke), Path('/tmp')))}")
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for exp_id in _selected_ids(args):
        meta = EXPERIMENTS[exp_id]
        missing = meta["check"](assets, bool(args.smoke))  # type: ignore[index]
        if missing:
            print(f"[blocked] {exp_id}: {'; '.join(missing)}")
            return 2
        cmd = meta["build"](assets, bool(args.smoke), out_dir)  # type: ignore[index]
        print(f"[run] {exp_id}")
        print(cmd_to_text(cmd))
        if args.dry_run:
            continue
        if not meta["run_supported"]:
            print(f"[manual] {exp_id} must be run manually.")
            continue
        assert isinstance(cmd, list)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
