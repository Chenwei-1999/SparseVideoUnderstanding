#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in {path}")
    return data


def _summary_path_from_command(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except Exception:
        return None
    for i, part in enumerate(parts):
        if part == "--summary-json" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _metric(results: dict[str, Any], name: str) -> Any:
    if name in results:
        return results[name]
    nested = results.get("results")
    if isinstance(nested, dict):
        return nested.get(name)
    return None


def _conversation_log_path(summary: dict[str, Any], fallback: Path) -> Path:
    raw_path = summary.get("log_jsonl") or summary.get("prompt_log_jsonl")
    nested = summary.get("results")
    if not raw_path and isinstance(nested, dict):
        raw_path = nested.get("log_jsonl") or nested.get("prompt_log_jsonl")
    path = Path(str(raw_path)) if raw_path else fallback
    if not path.is_absolute():
        path = fallback.parent / path
    return path


def _audit_conversation_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "lines": 0, "status": "missing_jsonl"}
    lines = 0
    has_prompt = False
    has_output = False
    has_messages = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lines += 1
            if lines > 200 and has_prompt and has_output:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("user_text") or obj.get("prompt") or obj.get("question"):
                has_prompt = True
            if obj.get("raw_output") or obj.get("output") or obj.get("prediction"):
                has_output = True
            messages = obj.get("messages") or obj.get("conversation") or obj.get("history")
            if isinstance(messages, list) and messages:
                has_messages = True
                has_prompt = True
                has_output = True
    if lines <= 0:
        status = "empty_jsonl"
    elif has_messages or (has_prompt and has_output):
        status = "ok"
    else:
        status = "missing_prompt_or_output"
    return {
        "path": str(path),
        "lines": lines,
        "status": status,
        "has_prompt": has_prompt,
        "has_output": has_output,
        "has_messages": has_messages,
    }


def _collect_experiment(exp: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    exp_id = str(exp.get("id") or "")
    command = str(exp.get("command") or "")
    explicit = _summary_path_from_command(command)
    summary_path = Path(explicit) if explicit else results_dir / f"{exp_id}.summary.json"
    if not summary_path.is_absolute():
        summary_path = results_dir / summary_path

    row: dict[str, Any] = {
        "id": exp_id,
        "job_id": exp.get("job_id"),
        "blocked": bool(exp.get("blocked")),
        "blocked_reasons": exp.get("blocked_reasons") or [],
        "summary_json": str(summary_path),
        "status": "blocked" if exp.get("blocked") else "missing_summary",
    }
    if row["blocked"] or not summary_path.exists():
        return row

    summary = _read_json(summary_path)
    log_audit = _audit_conversation_log(_conversation_log_path(summary, results_dir / f"{exp_id}.jsonl"))
    samples = _metric(summary, "samples")
    failed = _metric(summary, "failed")
    calls = _metric(summary, "total_model_calls")
    accuracy = _metric(summary, "accuracy")
    row.update(
        {
            "samples": samples,
            "failed": failed,
            "total_model_calls": calls,
            "accuracy": accuracy,
            "elapsed_s": _metric(summary, "elapsed_s"),
            "log_jsonl": log_audit["path"],
            "log_lines": log_audit["lines"],
            "conversation_log_status": log_audit["status"],
            "conversation_log_has_prompt": log_audit.get("has_prompt"),
            "conversation_log_has_output": log_audit.get("has_output"),
            "conversation_log_has_messages": log_audit.get("has_messages"),
        }
    )

    if calls is not None and int(calls) > 0 and log_audit["status"] != "ok":
        row["status"] = f"invalid_conversation_log:{log_audit['status']}"
    elif calls is not None and int(calls) > 0 and int(log_audit["lines"]) < int(calls):
        row["status"] = "invalid_conversation_log:missing_model_call_rows"
    elif calls is not None and int(calls) <= 0:
        row["status"] = "invalid_no_model_calls"
    elif samples is not None and int(samples) <= 0:
        row["status"] = "invalid_no_samples"
    elif failed is not None and samples is not None and int(failed) >= int(samples):
        row["status"] = "all_failed"
    else:
        row["status"] = "ok"
    return row


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "id",
        "job_id",
        "status",
        "samples",
        "failed",
        "total_model_calls",
        "accuracy",
        "log_lines",
        "conversation_log_status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = [row.get(h, "") for h in headers]
        lines.append("| " + " | ".join("" if v is None else str(v) for v in vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect paper-suite summary JSON files into one audit table.")
    parser.add_argument("run_dir", help="Run directory produced by submit_paper_suite_slurm.py.")
    parser.add_argument("--write", action="store_true", help="Write collected_summary.json and collected_summary.md.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    results_dir = Path(manifest.get("results_dir") or run_dir / "results")
    rows = [_collect_experiment(exp, results_dir) for exp in manifest.get("experiments", [])]

    payload = {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "rows": rows,
    }
    table = _markdown_table(rows)
    print(table)
    if args.write:
        (run_dir / "collected_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "collected_summary.md").write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
