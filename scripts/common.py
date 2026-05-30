from __future__ import annotations

import importlib.metadata
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.getenv("REVISE_ASSET_ROOT", REPO_ROOT / "data" / "revise_assets")).expanduser()


def _first_existing(candidates: list[str | Path]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def _env_or_existing(env_name: str, candidates: list[str | Path]) -> str | None:
    env_val = os.getenv(env_name, "").strip()
    if env_val and Path(env_val).expanduser().exists():
        return str(Path(env_val).expanduser())
    return _first_existing(candidates)


def _metadata_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        module_name = dist_name.replace("-", "_")
        try:
            module = __import__(module_name)
        except Exception:
            return None
        return getattr(module, "__version__", "installed")


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        return 1, str(exc)
    output = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, output


_PROBE_ROW_LIMIT = 32


def _read_json_array_prefix(path: str, limit: int) -> tuple[list[Any], bool]:
    """Parse up to *limit* elements from the front of a JSON-array file.

    The VideoEspresso annotation files are hundreds of megabytes, but the MC
    probe below only needs the first few rows. Loading the whole file with
    ``json.loads(read_text())`` costs roughly 3x its size in RAM (a multi-hundred
    MB file can balloon the process past a gigabyte) plus seconds of GPFS I/O.
    This reads the file incrementally with ``raw_decode`` and stops as soon as
    *limit* elements have been decoded, leaving the rest of the file unread (and
    tolerating a truncated/corrupt tail beyond the prefix).

    Returns ``(items, is_array)`` where ``is_array`` is True iff the first
    non-whitespace character is ``[``.
    """
    decoder = json.JSONDecoder()
    items: list[Any] = []
    buf = ""
    started = False
    with open(path, "r", encoding="utf-8") as f:
        while len(items) < limit:
            if not started:
                stripped = buf.lstrip()
                if not stripped:
                    chunk = f.read(65536)
                    if not chunk:
                        return items, False
                    buf = chunk
                    continue
                if stripped[0] != "[":
                    return items, False
                buf = stripped[1:]
                started = True
            buf = buf.lstrip().lstrip(",").lstrip()
            if buf[:1] == "]":
                return items, True
            if not buf:
                chunk = f.read(65536)
                if not chunk:
                    return items, True
                buf += chunk
                continue
            try:
                obj, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = f.read(65536)
                if not chunk:
                    return items, True
                buf += chunk
                continue
            items.append(obj)
            buf = buf[end:]
    return items, True


def _probe_videoespresso_mc(json_path: str | None) -> dict[str, Any]:
    out = {"path": json_path, "multiple_choice": False, "reason": "missing"}
    if not json_path or not Path(json_path).exists():
        return out
    try:
        rows, is_array = _read_json_array_prefix(json_path, _PROBE_ROW_LIMIT)
    except Exception as exc:
        out["reason"] = f"json_error: {type(exc).__name__}"
        return out
    if not is_array or not rows:
        out["reason"] = "empty_or_non_list"
        return out
    has_mc = True
    for row in rows:
        if not isinstance(row, dict):
            has_mc = False
            break
        options = row.get("options") or row.get("choices")
        answer = row.get("correct_answer")
        if not isinstance(options, list) or len(options) < 2 or answer in (None, ""):
            has_mc = False
            break
    out["multiple_choice"] = has_mc
    out["reason"] = "ok" if has_mc else "missing_options_or_correct_answer"
    return out


def _probe_nextqa_video(csv_path: str | None, map_json: str | None, video_root: str | None) -> dict[str, Any]:
    out = {"csv": csv_path, "ok": False, "reason": "missing"}
    if not csv_path or not map_json or not video_root:
        return out
    try:
        from revise.pnp_utils import normalize_video_id, resolve_nextqa_video_path

        with open(map_json, "r", encoding="utf-8") as f:
            video_map = {str(k): v for k, v in json.load(f).items()}
        checked = 0
        mapped = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                checked += 1
                video_id = normalize_video_id(row.get("video", ""))
                rel = video_map.get(video_id)
                if rel is not None:
                    mapped += 1
                if rel is not None:
                    video_path = resolve_nextqa_video_path(video_root, str(rel), video_id)
                    if video_path:
                        return {
                            "csv": csv_path,
                            "ok": True,
                            "reason": "ok",
                            "video_id": video_id,
                            "path": video_path,
                            "checked": checked,
                            "mapped": mapped,
                        }
                if checked >= 256:
                    break
        out.update({"reason": "no_resolvable_video_in_probe", "checked": checked, "mapped": mapped})
        return out
    except Exception as exc:
        out["reason"] = f"probe_error: {type(exc).__name__}"
        return out


def discover_assets() -> dict[str, Any]:
    nextqa_root = _env_or_existing(
        "REVISE_NEXTQA_ROOT",
        [ASSET_ROOT / "NExT-QA"],
    )
    nextqa_video_root = _env_or_existing(
        "REVISE_NEXTQA_VIDEO_ROOT",
        [Path(nextqa_root) / "NExTVideo"] if nextqa_root else [],
    )
    nextqa_map_json = _env_or_existing(
        "REVISE_NEXTQA_MAP_JSON",
        [Path(nextqa_root) / "map_vid_vidorID.json"] if nextqa_root else [],
    )
    nextqa_train_csv = _env_or_existing(
        "REVISE_NEXTQA_TRAIN_CSV",
        [Path(nextqa_root) / "nextqa" / "train.csv"] if nextqa_root else [],
    )
    nextqa_val_csv = _env_or_existing(
        "REVISE_NEXTQA_VAL_CSV",
        [Path(nextqa_root) / "nextqa" / "val.csv"] if nextqa_root else [],
    )
    nextqa_captions_dir = _env_or_existing(
        "REVISE_NEXTQA_CAPTIONS_DIR",
        [
            Path(nextqa_root) / "captions",
            Path(nextqa_root) / "captions_1fps",
            Path(nextqa_root) / "video_captions",
        ]
        if nextqa_root
        else [],
    )

    ve_root = _env_or_existing(
        "REVISE_VIDEOESPRESSO_ROOT",
        [ASSET_ROOT / "VideoEspresso"],
    )
    ve_test_json = _env_or_existing(
        "REVISE_VIDEOESPRESSO_TEST_JSON",
        [Path(ve_root) / "test_video" / "bench_hard.json"] if ve_root else [],
    )
    ve_test_video_root = _env_or_existing(
        "REVISE_VIDEOESPRESSO_TEST_VIDEO_ROOT",
        [Path(ve_root) / "test_video"] if ve_root else [],
    )
    ve_train_video_json = _env_or_existing(
        "REVISE_VIDEOESPRESSO_TRAIN_VIDEO_JSON",
        [Path(ve_root) / "train_video" / "videoespresso_train_video.json"] if ve_root else [],
    )
    ve_train_multi_json = _env_or_existing(
        "REVISE_VIDEOESPRESSO_TRAIN_MULTI_JSON",
        [Path(ve_root) / "train_multi_image" / "videoespresso_train.json"] if ve_root else [],
    )
    ve_mc_train_json = _env_or_existing(
        "REVISE_VIDEOESPRESSO_MC_TRAIN_JSON",
        [
            REPO_ROOT / "outputs" / "videoespresso_train_mc.json",
            Path(ve_root) / "train_video" / "videoespresso_train_mc.json",
            Path(ve_root) / "train_video" / "videoespresso_train_multiple_choice.json",
        ]
        if ve_root
        else [],
    )

    egoschema_video_root = _env_or_existing(
        "REVISE_EGOSCHEMA_VIDEO_ROOT",
        [
            ASSET_ROOT / "EgoSchema" / "videos",
        ],
    )
    egoschema_json = _env_or_existing(
        "REVISE_EGOSCHEMA_JSON",
        [
            ASSET_ROOT / "EgoSchema" / "pnp_subset_500.json",
        ],
    )

    video_cache_dir = _env_or_existing(
        "REVISE_VIDEO_CACHE_DIR",
        [ASSET_ROOT / "video_cache"],
    )

    qwen_3b = _env_or_existing(
        "REVISE_QWEN25_VL_3B_PATH",
        [
            ASSET_ROOT / "models" / "Qwen2.5-VL-3B-Instruct",
        ],
    )
    qwen_7b = _env_or_existing(
        "REVISE_QWEN25_VL_7B_PATH",
        [
            ASSET_ROOT / "models" / "Qwen2.5-VL-7B-Instruct",
        ],
    )
    qwen_72b = _env_or_existing(
        "REVISE_QWEN25_VL_72B_PATH",
        [ASSET_ROOT / "models" / "Qwen2.5-VL-72B-Instruct"],
    )
    qwen2_vl_7b = _env_or_existing(
        "REVISE_QWEN2_VL_7B_PATH",
        [ASSET_ROOT / "models" / "Qwen2-VL-7B-Instruct"],
    )
    internvl2_8b = _env_or_existing(
        "REVISE_INTERNVL2_8B_PATH",
        [ASSET_ROOT / "models" / "InternVL2-8B"],
    )
    llava_ov_7b = _env_or_existing(
        "REVISE_LLAVA_OV_7B_PATH",
        [ASSET_ROOT / "models" / "LLaVA-OneVision-Qwen2-7B-OV"],
    )
    llava_next_path = _env_or_existing(
        "REVISE_LLAVA_NEXT_PATH",
        [ASSET_ROOT / "third_party" / "LLaVA-NeXT"],
    )
    local_model = _env_or_existing("REVISE_LOCAL_MODEL_PATH", [])
    local_model_id = os.getenv("REVISE_LOCAL_MODEL_ID", "").strip() or None

    gpu_rc, gpu_out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )

    return {
        "repo_root": str(REPO_ROOT),
        "asset_root": str(ASSET_ROOT),
        "python": {"path": sys.executable, "version": sys.version.split()[0]},
        "packages": {
            "torch": _metadata_version("torch"),
            "transformers": _metadata_version("transformers"),
            "vllm": _metadata_version("vllm"),
            "sglang": _metadata_version("sglang"),
            "decord": _metadata_version("decord"),
            "imageio": _metadata_version("imageio"),
            "datasets": _metadata_version("datasets"),
            "hydra-core": _metadata_version("hydra-core"),
            "ray": _metadata_version("ray"),
            "wandb": _metadata_version("wandb"),
            "scikit-learn": _metadata_version("scikit-learn"),
        },
        "gpu": {"available": gpu_rc == 0, "raw": gpu_out},
        "models": {
            "local_model": local_model,
            "local_model_id": local_model_id,
            "qwen25_vl_3b": qwen_3b,
            "qwen25_vl_7b": qwen_7b,
            "qwen25_vl_72b": qwen_72b,
            "qwen2_vl_7b": qwen2_vl_7b,
            "internvl2_8b": internvl2_8b,
            "llava_ov_7b": llava_ov_7b,
            "llava_next_path": llava_next_path,
        },
        "remote_api": {
            "base_url": os.getenv("REVISE_API_BASE_URL", "").strip() or None,
            "model_id": os.getenv("REVISE_MODEL_ID", "").strip() or None,
            "api_key_present": bool(
                os.getenv("REVISE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY")
            ),
        },
        "datasets": {
            "nextqa": {
                "root": nextqa_root,
                "video_root": nextqa_video_root,
                "map_json": nextqa_map_json,
                "train_csv": nextqa_train_csv,
                "val_csv": nextqa_val_csv,
                "captions_dir": nextqa_captions_dir,
                "train_probe": _probe_nextqa_video(nextqa_train_csv, nextqa_map_json, nextqa_video_root),
                "val_probe": _probe_nextqa_video(nextqa_val_csv, nextqa_map_json, nextqa_video_root),
            },
            "videoespresso": {
                "root": ve_root,
                "test_json": ve_test_json,
                "test_video_root": ve_test_video_root,
                "train_video_json": ve_train_video_json,
                "train_multi_json": ve_train_multi_json,
                "mc_train_json": ve_mc_train_json,
                "public_train_probe": _probe_videoespresso_mc(ve_train_video_json),
                "mc_train_probe": _probe_videoespresso_mc(ve_mc_train_json),
            },
            "egoschema": {
                "video_root": egoschema_video_root,
                "json": egoschema_json,
            },
            "video_cache": {
                "root": video_cache_dir,
                "lvbench_dir": str(Path(video_cache_dir) / "lvbench") if video_cache_dir else None,
                "videomme_dir": str(Path(video_cache_dir) / "videomme") if video_cache_dir else None,
            },
        },
    }
