#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Optional

from scripts.common import REPO_ROOT


def cmd_to_text(cmd: list[str] | str) -> str:
    if isinstance(cmd, str):
        return cmd
    return shlex.join(cmd)


def _command_parts(cmd: list[str] | str) -> list[str]:
    if isinstance(cmd, list):
        return [str(part) for part in cmd]
    try:
        return shlex.split(str(cmd))
    except ValueError:
        return []


def _command_flag_value(cmd: list[str] | str, flag: str) -> Optional[str]:
    parts = _command_parts(cmd)
    for idx, part in enumerate(parts):
        if part == flag and idx + 1 < len(parts):
            return str(parts[idx + 1])
    return None


def _command_first_flag_value(cmd: list[str] | str, flags: tuple[str, ...]) -> Optional[str]:
    for flag in flags:
        value = _command_flag_value(cmd, flag)
        if value is not None:
            return value
    return None


def _normal_compare_value(value: object) -> str:
    raw = str(value)
    if not raw or raw.startswith("<"):
        return raw
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def _command_value_matches_summary(expected: object, observed: object) -> bool:
    if str(expected) == str(observed):
        return True
    return _normal_compare_value(expected) == _normal_compare_value(observed)


def _path_from_command_assignment(cmd: list[str] | str, key: str) -> Optional[Path]:
    parts = _command_parts(cmd)
    prefix = f"{key}="
    for part in parts:
        if part.startswith(prefix):
            path = Path(part.removeprefix(prefix))
            return path if path.is_absolute() else REPO_ROOT / path
    return None


def _summary_json_paths(cmd: list[str] | str) -> list[Path]:
    parts = _command_parts(cmd)
    paths: list[Path] = []
    for idx, part in enumerate(parts):
        if part == "--summary-json" and idx + 1 < len(parts):
            path = Path(parts[idx + 1])
            paths.append(path if path.is_absolute() else REPO_ROOT / path)

    trainer_dir = _path_from_command_assignment(cmd, "trainer.default_local_dir")
    if trainer_dir is not None:
        paths.extend(
            [
                trainer_dir / "summary.json",
                trainer_dir / "trainer_summary.json",
                trainer_dir / "validation_summary.json",
                trainer_dir / "eval_summary.json",
                trainer_dir / "metrics.json",
            ]
        )
        if trainer_dir.is_dir():
            paths.extend(sorted(trainer_dir.glob("*summary*.json")))
            paths.extend(sorted(trainer_dir.glob("*metrics*.json")))
    return paths


def _summary_metric(summary: dict[str, object], name: str) -> object:
    if name in summary:
        return summary[name]
    nested = summary.get("results")
    if isinstance(nested, dict):
        return nested.get(name)
    return None


def _summary_setting(summary: dict[str, object], name: str) -> object:
    value = _summary_metric(summary, name)
    if value is not None:
        return value
    config = summary.get("config")
    if isinstance(config, dict):
        return config.get(name)
    return None


def _float_metric(summary: dict[str, object], name: str) -> Optional[float]:
    value = _summary_metric(summary, name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_value_matches_command_flag(
    summary: dict[str, object],
    cmd: list[str] | str,
    summary_name: str,
    flags: tuple[str, ...],
    *,
    required: bool,
) -> bool:
    expected = _command_first_flag_value(cmd, flags)
    if expected is None:
        return True
    observed = _summary_setting(summary, summary_name)
    if observed is None:
        return not required
    return _command_value_matches_summary(expected, observed)


def _is_nextqa_table4_rl_command(cmd: list[str] | str) -> bool:
    return any(str(part).endswith("run_nextqa_table4_rl_after_sft.py") for part in _command_parts(cmd))


def _expected_rl_checkpoint_path(cmd: list[str] | str) -> Optional[Path]:
    checkpoint_dir = _command_flag_value(cmd, "--checkpoint-dir")
    if checkpoint_dir is None:
        return None
    step = _command_flag_value(cmd, "--steps") or "100"
    return Path(_normal_compare_value(checkpoint_dir)) / f"global_step_{step}" / "actor" / "huggingface"


def _summary_path_setting_manifest(summary_path: Path) -> Path:
    if summary_path.name.endswith(".summary.json"):
        return summary_path.with_name(summary_path.name.replace(".summary.json", ".settings.json"))
    return summary_path.with_suffix(".settings.json")


def _manifest_int_matches_flag(settings: dict[str, object], cmd: list[str] | str, key: str, flag: str) -> bool:
    expected = _command_flag_value(cmd, flag)
    if expected is None:
        return False
    try:
        return int(settings.get(key)) == int(expected)
    except (TypeError, ValueError):
        return False


def _manifest_float_matches_flag(settings: dict[str, object], cmd: list[str] | str, key: str, flag: str) -> bool:
    expected = _command_flag_value(cmd, flag)
    if expected is None:
        return False
    try:
        return abs(float(settings.get(key)) - float(expected)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _rl_setting_manifest_matches_command(cmd: list[str] | str, summary_path: Path) -> bool:
    settings_path = _summary_path_setting_manifest(summary_path)
    if not settings_path.is_file():
        return False
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(settings, dict):
        return False
    if settings.get("setting") != "nextqa_table4_rl_after_sft":
        return False
    expected_sft = _command_flag_value(cmd, "--sft-path")
    observed_sft = settings.get("sft_path")
    if expected_sft is None or observed_sft is None or not _command_value_matches_summary(expected_sft, observed_sft):
        return False
    expected_checkpoint_dir = _command_flag_value(cmd, "--checkpoint-dir")
    observed_checkpoint_dir = settings.get("checkpoint_dir")
    if (
        expected_checkpoint_dir is None
        or observed_checkpoint_dir is None
        or not _command_value_matches_summary(expected_checkpoint_dir, observed_checkpoint_dir)
    ):
        return False
    for key, flag in (
        ("steps", "--steps"),
        ("n_gpus", "--n-gpus"),
        ("rollout_tensor_parallel_size", "--rollout-tensor-parallel-size"),
        ("eval_tensor_parallel_size", "--eval-tensor-parallel-size"),
        ("train_batch_size", "--train-batch-size"),
        ("ppo_mini_batch_size", "--ppo-mini-batch-size"),
        ("rollout_n", "--rollout-n"),
        ("max_rounds", "--max-rounds"),
        ("max_frames_per_round", "--max-frames-per-round"),
        ("max_retries_per_round", "--max-retries-per-round"),
        ("min_select_rounds", "--min-select-rounds"),
    ):
        if not _manifest_int_matches_flag(settings, cmd, key, flag):
            return False
    if settings.get("use_1fps_timeline") is not True:
        return False
    reward = settings.get("reward")
    if not isinstance(reward, dict):
        return False
    for key, flag in (
        ("lambda_conf", "--lambda-conf"),
        ("lambda_sum", "--lambda-sum"),
        ("lambda_stop", "--lambda-stop"),
        ("gamma", "--gamma"),
        ("format_reward", "--format-reward"),
        ("stop_round_threshold", "--stop-round-threshold"),
        ("stop_bonus_beta", "--stop-bonus-beta"),
    ):
        if not _manifest_float_matches_flag(reward, cmd, key, flag):
            return False
    return True


def _is_nextqa_table4_sft_summary(cmd: list[str] | str, summary_path: Path) -> bool:
    summary_json = _command_flag_value(cmd, "--summary-json")
    return summary_path.name == "nextqa_table4_sft.summary.json" or (
        summary_json is not None and Path(summary_json).name == "nextqa_table4_sft.summary.json"
    )


def _table4_sft_provenance_paths(cmd: list[str] | str, summary_path: Path) -> list[Path]:
    candidates = [_summary_path_setting_manifest(summary_path)]
    env_path = os.getenv("REVISE_NEXTQA_SFT_PROVENANCE", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    model_path = _command_flag_value(cmd, "--model-path")
    if model_path:
        checkpoint_path = Path(_normal_compare_value(model_path))
        candidates.extend(
            [
                checkpoint_path / "revise_sft_provenance.json",
                checkpoint_path.parent / "revise_sft_provenance.json",
                checkpoint_path.parent.parent / "revise_sft_provenance.json",
            ]
        )
    unique = []
    seen = set()
    for path in candidates:
        if not path.is_absolute():
            path = REPO_ROOT / path
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _table4_sft_provenance_matches_command(cmd: list[str] | str, summary_path: Path) -> bool:
    expected_sft_path = _command_flag_value(cmd, "--model-path")
    if expected_sft_path is None:
        return False
    for path in _table4_sft_provenance_paths(cmd, summary_path):
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        if manifest.get("setting") != "nextqa_table4_sft" or manifest.get("dataset") != "nextqa":
            continue
        observed_sft_path = manifest.get("sft_path") or manifest.get("checkpoint_path") or manifest.get("model_path")
        if observed_sft_path is None or not _command_value_matches_summary(expected_sft_path, observed_sft_path):
            continue
        sft_generate = manifest.get("sft_generate")
        if not isinstance(sft_generate, dict):
            continue
        try:
            max_rounds = int(sft_generate.get("max_rounds"))
            min_first_select_ratio = float(sft_generate.get("min_first_select_ratio"))
        except (TypeError, ValueError):
            continue
        if max_rounds != 4 or abs(min_first_select_ratio - 0.45) > 1e-9:
            continue
        if sft_generate.get("variable_length_traces") is not True:
            continue
        return True
    return False


def _summary_matches_command(summary: dict[str, object], cmd: list[str] | str, summary_path: Path) -> bool:
    for flag, setting in (
        ("--max-samples", "max_samples"),
        ("--max-frames", "max_frames"),
        ("--max-rounds", "max_rounds"),
        ("--max-frames-per-round", "max_frames_per_round"),
        ("--max-retries-per-round", "max_retries_per_round"),
        ("--min-select-rounds", "min_select_rounds"),
    ):
        expected = _command_flag_value(cmd, flag)
        if expected is None:
            continue
        observed = _summary_setting(summary, setting)
        if observed is None:
            return False
        try:
            if int(observed) != int(expected):
                return False
        except (TypeError, ValueError):
            return False
    for summary_name, flags in (
        ("dataset_csv", ("--csv", "--val-csv")),
        ("video_root", ("--video-root",)),
        ("map_json", ("--map-json",)),
        ("model_path", ("--model-path",)),
    ):
        if not _summary_value_matches_command_flag(summary, cmd, summary_name, flags, required=True):
            return False
    if _is_nextqa_table4_sft_summary(cmd, summary_path):
        if not _table4_sft_provenance_matches_command(cmd, summary_path):
            return False
    if _is_nextqa_table4_rl_command(cmd):
        expected_checkpoint = _expected_rl_checkpoint_path(cmd)
        observed_checkpoint = _summary_setting(summary, "checkpoint_path") or _summary_setting(summary, "model_path")
        if expected_checkpoint is None or observed_checkpoint is None:
            return False
        if not _command_value_matches_summary(str(expected_checkpoint), observed_checkpoint):
            return False
        if not _rl_setting_manifest_matches_command(cmd, summary_path):
            return False
    return True


def _observed_metrics_from_summary(cmd: list[str] | str) -> Optional[dict[str, float]]:
    summary: dict[str, object] | None = None
    seen_paths: set[Path] = set()
    for summary_path in _summary_json_paths(cmd):
        if summary_path in seen_paths:
            continue
        seen_paths.add(summary_path)
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path, encoding="utf-8") as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict) and _summary_matches_command(loaded, cmd, summary_path):
            summary = loaded
            break
    if summary is None:
        return None

    accuracy = _float_metric(summary, "accuracy")
    frames = _float_metric(summary, "avg_frames_used")
    if frames is None:
        frames = _float_metric(summary, "avg_frames")
    rounds = _float_metric(summary, "avg_rounds")
    if rounds is None and str(summary.get("task") or "").startswith("oneshot"):
        rounds = 1.0
    elapsed_s = _float_metric(summary, "elapsed_s")
    samples = _float_metric(summary, "samples")

    metrics: dict[str, float] = {}
    if accuracy is not None:
        metrics["acc_pct"] = round(accuracy * 100.0 if 0.0 <= accuracy <= 1.0 else accuracy, 6)
    if frames is not None:
        metrics["frames"] = round(frames, 6)
    if rounds is not None:
        metrics["rounds"] = round(rounds, 6)
    if elapsed_s is not None and samples and samples > 0:
        metrics["time_s"] = round(elapsed_s / samples, 6)
    return metrics or None


def _verification_status(observed_metrics: Optional[dict[str, float]], missing: list[str]) -> str:
    if observed_metrics is None:
        return "not_run"
    if missing:
        return "observed_with_blockers"
    return "observed"


def _paper_target_delta(
    observed_metrics: Optional[dict[str, float]], paper_metrics: object
) -> Optional[dict[str, float]]:
    if observed_metrics is None or not isinstance(paper_metrics, dict):
        return None

    deltas: dict[str, float] = {}
    for key in ("acc_pct", "frames", "rounds", "time_s"):
        observed = observed_metrics.get(key)
        target = paper_metrics.get(key)
        if observed is None or target is None:
            continue
        try:
            deltas[key] = round(float(observed) - float(target), 6)
        except (TypeError, ValueError):
            continue
    return deltas or None


def _paper_comparison_status(
    observed_metrics: Optional[dict[str, float]], paper_metrics: object, missing: list[str]
) -> str:
    if observed_metrics is None:
        return "not_observed"
    if missing:
        return "diagnostic_only"
    deltas = _paper_target_delta(observed_metrics, paper_metrics)
    if deltas is None:
        return "observed_no_paper_target"
    if any(abs(value) > 1e-6 for value in deltas.values()):
        return "observed_with_target_gap"
    return "matches_reported_target"


def _paper_reproduction_status(experiments: list[dict[str, object]]) -> str:
    if any(row["readiness_status"] == "blocked" for row in experiments):
        return "blocked"
    statuses = [str(row["paper_comparison_status"]) for row in experiments]
    if any(status == "observed_with_target_gap" for status in statuses):
        return "observed_with_target_gap"
    if statuses and all(status == "matches_reported_target" for status in statuses):
        return "matches_reported_target"
    if any(status.startswith("observed") for status in statuses):
        return "partially_observed"
    return "not_run"


def _suite_verification_status(experiments: list[dict[str, object]]) -> str:
    has_observed = any(row["verification_status"] == "observed" for row in experiments)
    has_observed_with_blockers = any(row["verification_status"] == "observed_with_blockers" for row in experiments)
    has_blockers = any(row["readiness_status"] == "blocked" for row in experiments)
    if has_observed_with_blockers or (has_observed and has_blockers):
        return "observed_with_blockers"
    if has_observed:
        return "observed"
    return "not_run"


def build_report(
    experiments_catalog: dict[str, dict[str, object]],
    exp_ids: list[str],
    assets: dict,
    smoke: bool,
    out_dir: Path,
) -> dict[str, object]:
    experiments: list[dict[str, object]] = []
    for exp_id in exp_ids:
        meta = experiments_catalog[exp_id]
        missing = meta["check"](assets, smoke)  # type: ignore[index]
        command = meta["build"](assets, smoke, out_dir)  # type: ignore[index]
        readiness_status = "ready" if not missing else "blocked"
        observed_metrics = _observed_metrics_from_summary(command)
        verification_status = _verification_status(observed_metrics, missing)
        paper_target_metrics = meta.get("paper_metrics")
        paper_target_delta = _paper_target_delta(observed_metrics, paper_target_metrics)
        paper_comparison_status = _paper_comparison_status(observed_metrics, paper_target_metrics, missing)
        experiments.append(
            {
                "id": exp_id,
                "title": meta["title"],
                "paper_ref": meta["paper_ref"],
                "paper_metrics_source": meta["paper_ref"],
                "paper_target_metrics": paper_target_metrics,
                "paper_target_delta": paper_target_delta,
                "paper_comparison_status": paper_comparison_status,
                "setting_note": meta.get("setting_note"),
                "observed_metrics": observed_metrics,
                "verification_status": verification_status,
                "readiness_status": readiness_status,
                "missing": missing,
                "run_supported": bool(meta["run_supported"]),
                "command": command,
                "command_text": cmd_to_text(command),
            }
        )
    return {
        "suite": "paper_suite",
        "smoke": bool(smoke),
        "output_dir": str(out_dir),
        "readiness_status": "ready" if all(not row["missing"] for row in experiments) else "blocked",
        "paper_reproduction_status": _paper_reproduction_status(experiments),
        "verification_status": _suite_verification_status(experiments),
        "experiments": experiments,
    }


def _report_metric_items(metrics: object, *, signed: bool = False) -> list[str]:
    if not isinstance(metrics, dict):
        return []
    items = []
    for key, label in (
        ("acc_pct", "acc"),
        ("frames", "frames"),
        ("rounds", "rounds"),
        ("time_s", "time"),
    ):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        sign = "+" if signed else ""
        items.append(f"{label}={value_f:{sign}.2f}")
    return items


def report_summary_lines(report: dict[str, object], output_json: Path | None = None) -> list[str]:
    lines = []
    if output_json is not None:
        lines.append(f"wrote JSON report: {output_json}")
    lines.append(
        "suite: "
        f"readiness={report.get('readiness_status')} "
        f"verification={report.get('verification_status')} "
        f"paper_reproduction={report.get('paper_reproduction_status')}"
    )
    experiments = report.get("experiments")
    if not isinstance(experiments, list):
        return lines
    for row in experiments:
        if not isinstance(row, dict):
            continue
        status = (
            f"{row.get('readiness_status')}/"
            f"{row.get('verification_status')}/"
            f"{row.get('paper_comparison_status')}"
        )
        line = f"[{status}] {row.get('id')}"
        observed_items = _report_metric_items(row.get("observed_metrics"))
        if observed_items:
            line += " observed " + " ".join(observed_items)
        delta_items = _report_metric_items(row.get("paper_target_delta"), signed=True)
        if delta_items:
            line += " delta " + " ".join(delta_items)
        lines.append(line)
        missing = row.get("missing")
        if isinstance(missing, list) and missing:
            for item in missing:
                lines.append(f"  blocker: {item}")
    return lines
