#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPERIMENTS = {
    "nextqa_train_pipeline": {
        "teacher_stems": ("nextqa_teacher_smoke", "nextqa_teacher"),
        "sft_prefix": "nextqa_revise_sft",
        "sft_ckpt": "nextqa_sft",
        "rl_ckpt": "nextqa_grpo_after_sft",
    },
    "videoespresso_train_pipeline": {
        "teacher_stems": ("videoespresso_teacher_smoke", "videoespresso_teacher"),
        "sft_prefix": "videoespresso_revise_sft",
        "sft_ckpt": "videoespresso_sft",
        "rl_ckpt": "videoespresso_grpo_after_sft",
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _find_first(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        pos = text.find("{", idx)
        if pos < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            idx = pos + 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        idx = pos + max(end, 1)
    return out


def _teacher_summary_from_stdout(log_dir: Path, job_id: str | None, experiment: str) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if job_id:
        candidates.extend(sorted(log_dir.glob(f"*{job_id}.out")))
    candidates.extend(sorted(log_dir.glob(f"*{experiment}*.out")))
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for obj in _extract_json_objects(text):
            if "total_model_calls" in obj and "samples" in obj:
                return obj
    return None


def _dir_has_files(path: Path) -> bool:
    return path.exists() and any(child.is_file() for child in path.rglob("*"))


def _summarize(run_dir: Path, experiment: str, job_id: str | None) -> dict[str, Any]:
    spec = EXPERIMENTS[experiment]
    results_dir = run_dir / "results"
    teacher_log = _find_first(
        [results_dir / "teacher_logs" / f"{stem}.jsonl" for stem in spec["teacher_stems"]]
    )
    server_log = _find_first(
        [results_dir / "server_logs" / f"{stem}.server.log" for stem in spec["teacher_stems"]]
    )
    sft_train = results_dir / "sft_data" / f"{spec['sft_prefix']}_train.parquet"
    sft_val = results_dir / "sft_data" / f"{spec['sft_prefix']}_val.parquet"
    sft_ckpt = results_dir / "checkpoints" / str(spec["sft_ckpt"])
    rl_ckpt = results_dir / "checkpoints" / str(spec["rl_ckpt"])

    teacher_rows = _read_jsonl(teacher_log) if teacher_log else []
    sample_ids = {str(row.get("sample_id")) for row in teacher_rows if row.get("sample_id") is not None}
    teacher_summary = (
        _teacher_summary_from_stdout(run_dir / "logs", job_id, experiment)
        or _teacher_summary_from_stdout(run_dir / "slurm", job_id, experiment)
        or {}
    )
    samples = int(teacher_summary.get("samples") or len(sample_ids))
    calls = int(teacher_summary.get("total_model_calls") or len(teacher_rows))
    failed = int(teacher_summary.get("failed") or 0)
    required = {
        "teacher_log": bool(teacher_log and teacher_log.exists() and teacher_log.stat().st_size > 0),
        "server_log": bool(server_log and server_log.exists() and server_log.stat().st_size > 0),
        "sft_train_parquet": bool(sft_train.exists() and sft_train.stat().st_size > 0),
        "sft_val_parquet": bool(sft_val.exists() and sft_val.stat().st_size > 0),
        "sft_checkpoint": _dir_has_files(sft_ckpt),
        "rl_checkpoint": _dir_has_files(rl_ckpt),
    }
    complete = all(required.values())
    return {
        "task": experiment,
        "training_pipeline": True,
        "samples": samples,
        "failed": failed if complete else max(failed, samples or 1),
        "total_model_calls": calls,
        "accuracy": teacher_summary.get("accuracy"),
        "elapsed_s": teacher_summary.get("elapsed_s"),
        "log_jsonl": str(teacher_log) if teacher_log else "",
        "prompt_log_jsonl": str(teacher_log) if teacher_log else "",
        "teacher_summary": teacher_summary,
        "stages": {
            "complete": complete,
            "required": required,
            "teacher_log": str(teacher_log) if teacher_log else "",
            "server_log": str(server_log) if server_log else "",
            "sft_train_parquet": str(sft_train),
            "sft_val_parquet": str(sft_val),
            "sft_checkpoint": str(sft_ckpt),
            "rl_checkpoint": str(rl_ckpt),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a collector-compatible summary for training pipeline runs.")
    parser.add_argument("run_dir")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--job-id", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary = _summarize(run_dir, args.experiment, args.job_id or None)
    out = Path(args.out).expanduser() if args.out else run_dir / "results" / f"{args.experiment}.summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out), "complete": summary["stages"]["complete"]}, ensure_ascii=False, indent=2))
    return 0 if summary["stages"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
