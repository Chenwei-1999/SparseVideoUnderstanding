#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_run_summaries import _collect_experiment, _metric


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _iter_jsonl(path: Path):
    if not path.exists():
        return
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
                yield obj


def _summary_path(exp: dict[str, Any], results_dir: Path) -> Path:
    row = _collect_experiment(exp, results_dir)
    return Path(str(row["summary_json"]))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_pct(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return ""
    return f"{100.0 * v:.2f}"


def _format_float(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return ""
    return f"{v:.2f}"


def _metric_any(summary: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = _metric(summary, name)
        if value is not None:
            return value
    return None


def _method_label(exp_id: str) -> str:
    labels = {
        "nextqa_pnp": "NExT-QA / ReViSe PNP",
        "nextqa_oneshot": "NExT-QA / one-shot",
        "nextqa_caption": "NExT-QA / caption baseline",
        "nextqa_videoagent": "NExT-QA / VideoAgent",
        "nextqa_videoagent_official": "NExT-QA / VideoAgent official-style",
        "videoespresso_pnp": "VideoEspresso / ReViSe PNP",
        "videoespresso_oneshot": "VideoEspresso / one-shot",
        "egoschema_pnp": "EgoSchema / ReViSe PNP",
        "videomme_pnp": "Video-MME / ReViSe PNP",
        "videomme_oneshot": "Video-MME / one-shot",
        "lvbench_pnp": "LVBench / ReViSe PNP",
        "lvbench_oneshot": "LVBench / one-shot",
    }
    return labels.get(exp_id, exp_id)


def _sample_key(obj: dict[str, Any]) -> str | None:
    for key in ("sample_id", "qid", "uid", "question_id"):
        value = obj.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _category(obj: dict[str, Any], exp_id: str) -> str:
    for key in ("task", "category", "domain", "dataset"):
        value = obj.get(key)
        if value:
            return str(value)
    return exp_id


def _answer_letter(value: Any, options: list[Any] | None = None) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return chr(ord("A") + value)
    text = str(value).strip().upper()
    if len(text) == 1 and "A" <= text <= "Z":
        return text
    if text.isdigit():
        idx = int(text)
        # NExT-QA stores labels as 0-based indices.
        if options and 0 <= idx < len(options):
            return chr(ord("A") + idx)
    return None


def _jsonl_category_rows(exp_id: str, log_path: Path) -> list[dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for obj in _iter_jsonl(log_path) or []:
        key = _sample_key(obj)
        if key is None:
            continue
        rec = samples.setdefault(key, {"sample_id": key, "category": _category(obj, exp_id)})
        if rec.get("category") == exp_id:
            rec["category"] = _category(obj, exp_id)
        options = obj.get("options") or obj.get("choices")
        if isinstance(options, list):
            rec["options"] = options
        gt = obj.get("answer_gt")
        if gt is None:
            gt = obj.get("ground_truth_idx")
        if gt is not None:
            rec["gt"] = _answer_letter(gt, rec.get("options"))
        pred = obj.get("final_answer")
        if pred is None:
            pred = obj.get("answer_letter")
        if pred is not None:
            rec["pred"] = _answer_letter(pred, rec.get("options"))

    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"samples": 0, "correct": 0, "answered": 0})
    for rec in samples.values():
        cat = str(rec.get("category") or exp_id)
        gt = rec.get("gt")
        pred = rec.get("pred")
        by_cat[cat]["samples"] += 1
        if pred is not None:
            by_cat[cat]["answered"] += 1
        if gt is not None and pred is not None and gt == pred:
            by_cat[cat]["correct"] += 1

    rows = []
    for cat, vals in sorted(by_cat.items()):
        denom = max(1, vals["answered"])
        rows.append(
            {
                "experiment": exp_id,
                "category": cat,
                "samples": vals["samples"],
                "answered": vals["answered"],
                "correct": vals["correct"],
                "accuracy": vals["correct"] / denom,
            }
        )
    return rows


def _collect_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _read_json(run_dir / "manifest.json")
    results_dir = Path(manifest.get("results_dir") or run_dir / "results")
    overall = []
    by_category = []
    for exp in manifest.get("experiments", []):
        row = _collect_experiment(exp, results_dir)
        if row.get("status") != "ok":
            overall.append(
                {
                    "run_dir": str(run_dir),
                    "experiment": row.get("id"),
                    "method": _method_label(str(row.get("id") or "")),
                    "status": row.get("status"),
                    "job_id": row.get("job_id"),
                    "summary_json": row.get("summary_json"),
                    "log_jsonl": row.get("log_jsonl"),
                }
            )
            continue
        summary_path = _summary_path(exp, results_dir)
        summary = _read_json(summary_path)
        exp_id = str(row.get("id") or exp.get("id") or "")
        log_jsonl = Path(str(row.get("log_jsonl") or summary.get("log_jsonl") or summary.get("prompt_log_jsonl") or ""))
        overall.append(
            {
                "run_dir": str(run_dir),
                "experiment": exp_id,
                "method": _method_label(exp_id),
                "status": "ok",
                "job_id": row.get("job_id"),
                "samples": _metric_any(summary, "samples"),
                "answered": _metric_any(summary, "answered"),
                "correct": _metric_any(summary, "correct"),
                "failed": _metric_any(summary, "failed"),
                "accuracy": _metric_any(summary, "accuracy"),
                "avg_rounds": _metric_any(summary, "avg_rounds"),
                "avg_frames_used": _metric_any(summary, "avg_frames_used", "avg_frames"),
                "total_model_calls": _metric_any(summary, "total_model_calls"),
                "summary_json": str(summary_path),
                "log_jsonl": str(log_jsonl),
            }
        )
        if str(log_jsonl):
            by_category.extend(_jsonl_category_rows(exp_id, log_jsonl))
    return overall, by_category


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for header in headers:
            value = row.get(header, "")
            if header == "accuracy":
                value = _format_pct(value)
            elif header in {"avg_rounds", "avg_frames_used"}:
                value = _format_float(value)
            vals.append("" if value is None else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build verified result tables from completed paper-suite runs.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories produced by paper_suite.py or a launcher.")
    parser.add_argument("--out-json", default="", help="Optional output JSON path.")
    parser.add_argument("--out-md", default="", help="Optional output Markdown path.")
    args = parser.parse_args()

    overall: list[dict[str, Any]] = []
    by_category: list[dict[str, Any]] = []
    for run in args.run_dirs:
        run_overall, run_categories = _collect_run(Path(run).resolve())
        overall.extend(run_overall)
        by_category.extend(run_categories)

    payload = {"overall": overall, "by_category": by_category}
    md = (
        "# Verified Result Snapshot\n\n"
        "## Overall\n\n"
        + _markdown_table(
            overall,
            [
                "experiment",
                "job_id",
                "status",
                "samples",
                "answered",
                "correct",
                "failed",
                "accuracy",
                "avg_rounds",
                "avg_frames_used",
                "total_model_calls",
                "summary_json",
                "log_jsonl",
            ],
        )
        + "\n\n## By Category\n\n"
        + _markdown_table(by_category, ["experiment", "category", "samples", "answered", "correct", "accuracy"])
        + "\n"
    )

    print(md)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
