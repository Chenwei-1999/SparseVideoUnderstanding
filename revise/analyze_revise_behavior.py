#!/usr/bin/env python3
"""Summarize REVISE action/round behavior for setting sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise.generate_sft_data import first_assistant_action  # noqa: E402
from revise.pnp.utils import parse_strict_revise_action  # noqa: E402


def _counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: str(item))}


def _action_from_output(raw_output: str) -> str:
    parsed = parse_strict_revise_action(raw_output or "")
    if parsed is None:
        return "invalid"
    return str(parsed["kind"])


def _messages_from_cell(cell: Any) -> list[dict[str, Any]]:
    if isinstance(cell, list):
        return [dict(item) for item in cell if isinstance(item, dict)]
    if hasattr(cell, "tolist"):
        value = cell.tolist()
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def analyze_sft_parquet(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    assistant_turns: Counter[int] = Counter()
    first_actions: Counter[str] = Counter()
    action_sequences: Counter[str] = Counter()

    for cell in df.get("messages", []):
        messages = _messages_from_cell(cell)
        actions = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            actions.append(_action_from_output(str(message.get("content") or "")))
        assistant_turns[len(actions)] += 1
        first_actions[first_assistant_action(messages)] += 1
        action_sequences["->".join(actions) if actions else "none"] += 1

    rows = int(len(df))
    first_select = first_actions.get("select", 0)
    return {
        "kind": "sft_parquet",
        "path": str(path),
        "rows": rows,
        "assistant_turns": _counter_json(assistant_turns),
        "first_actions": _counter_json(first_actions),
        "first_select_ratio": float(first_select / rows) if rows else 0.0,
        "top_action_sequences": dict(action_sequences.most_common(10)),
    }


def analyze_eval_jsonl(path: Path, *, max_rounds: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_sample = 0
    action_rows: Counter[str] = Counter()
    invalid_rows = 0
    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or f"__row_{idx}")
        if "sample_id" not in row:
            missing_sample += 1
        by_sample[sample_id].append(row)
        action = _action_from_output(str(row.get("raw_output") or ""))
        action_rows[action] += 1
        if action == "invalid":
            invalid_rows += 1

    final_rounds: Counter[int] = Counter()
    select_counts: Counter[int] = Counter()
    answered = 0
    correct = 0
    no_answer = 0

    for sample_rows in by_sample.values():
        sample_rows.sort(key=lambda row: (int(row.get("round_idx") or 0), int(row.get("retry_idx") or 0)))
        seen_selects = 0
        found_answer = False
        for row in sample_rows:
            action = _action_from_output(str(row.get("raw_output") or ""))
            if action == "select":
                seen_selects += 1
                continue
            answer = row.get("answer_letter") or row.get("final_answer")
            if action == "answer" or answer:
                answered += 1
                try:
                    round_idx = int(row.get("round_idx") or max_rounds)
                except Exception:
                    round_idx = max_rounds
                round_idx = min(max(round_idx, 1), int(max_rounds))
                final_rounds[round_idx] += 1
                select_counts[seen_selects] += 1
                try:
                    gold_idx = int(row.get("ground_truth_idx", -1))
                except Exception:
                    gold_idx = -1
                pred = str(answer or "").strip().upper()
                if gold_idx >= 0 and pred == chr(ord("A") + gold_idx):
                    correct += 1
                found_answer = True
                break
        if not found_answer:
            no_answer += 1

    total_final_rounds = sum(round_idx * count for round_idx, count in final_rounds.items())
    total_selects = sum(num_selects * count for num_selects, count in select_counts.items())
    return {
        "kind": "eval_jsonl",
        "path": str(path),
        "rows": len(rows),
        "samples": len(by_sample),
        "missing_sample_rows": missing_sample,
        "answered": answered,
        "no_answer": no_answer,
        "correct": correct,
        "accuracy": float(correct / answered) if answered else 0.0,
        "avg_final_round": float(total_final_rounds / answered) if answered else 0.0,
        "avg_selects_before_answer": float(total_selects / answered) if answered else 0.0,
        "final_round_distribution": _counter_json(final_rounds),
        "selects_before_answer_distribution": _counter_json(select_counts),
        "row_actions": _counter_json(action_rows),
        "invalid_rows": invalid_rows,
    }


def analyze_summary_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") if isinstance(data, dict) else None
    return {
        "kind": "summary_json",
        "path": str(path),
        "results": results if isinstance(results, dict) else {},
    }


def analyze_path(path: Path, *, max_rounds: int) -> dict[str, Any]:
    if path.suffix == ".parquet":
        return analyze_sft_parquet(path)
    if path.suffix == ".jsonl":
        return analyze_eval_jsonl(path, max_rounds=max_rounds)
    if path.suffix == ".json":
        return analyze_summary_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    reports = [analyze_path(path, max_rounds=args.max_rounds) for path in args.paths]
    payload: Any = reports[0] if len(reports) == 1 else reports
    print(json.dumps(payload, indent=args.indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
