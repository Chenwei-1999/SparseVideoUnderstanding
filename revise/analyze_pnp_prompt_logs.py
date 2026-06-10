#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from typing import Any, Optional

_SUMMARIZE_RE = re.compile(r"<summarize>(.*?)</summarize>", re.DOTALL | re.IGNORECASE)
_SELECT_RE = re.compile(r"<select>(.*?)</select>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_L_RE = re.compile(r"Total frames L = (\d+)")


def _extract_tag(text: str, pattern: re.Pattern[str]) -> Optional[str]:
    m = list(pattern.finditer(text or ""))
    if not m:
        return None
    return m[-1].group(1).strip()


def _parse_frame_indices(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text or "")]


def _normalize_answer_letter(answer_text: str, num_choices: int) -> Optional[str]:
    allowed = {chr(ord("A") + i) for i in range(max(0, num_choices))}
    if not allowed:
        allowed = {"A", "B", "C", "D", "E"}

    candidate = (answer_text or "").strip().upper()
    if candidate in allowed:
        return candidate
    m = re.search(r"\b([A-E])\b", candidate)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    m = re.search(r"([A-E])", candidate)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    return None


def _extract_frame_count(user_text: str) -> int:
    m = _L_RE.search(user_text or "")
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _safe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def iter_log_lines(paths: list[str]) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    objs.append(json.loads(line))
                except Exception:
                    continue
    return objs


def _counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: str(item))}


def _group_prompt_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if any(row.get("sample_id") for row in rows):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for idx, row in enumerate(rows):
            sample_id = str(row.get("sample_id") or f"__missing_sample_id_{idx}")
            qid = str(row.get("qid") or "")
            group_key = f"{sample_id}::{qid}" if qid else sample_id
            grouped[group_key].append(row)
        return [
            sorted(
                sample_rows,
                key=lambda row: (
                    int(row.get("round_idx") or 0),
                    int(row.get("retry_idx") or 0),
                    float(row.get("ts") or 0.0),
                ),
            )
            for sample_rows in grouped.values()
        ]

    # Legacy logs did not always include a stable sample id. Preserve the old
    # round-reset boundary inference for that format only.
    sorted_rows = sorted(rows, key=lambda row: float(row.get("ts") or 0.0))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in sorted_rows:
        forced = bool(row.get("forced_answer", False))
        round_idx = int(row.get("round_idx") or 0)
        if current and round_idx == 1 and not forced:
            groups.append(current)
            current = [row]
        else:
            current.append(row)
    if current:
        groups.append(current)
    return groups


def analyze_prompt_rows(rows: list[dict[str, Any]], *, max_rounds: int) -> dict[str, Any]:
    samples = 0
    answered = 0
    correct = 0

    term_reasons: Counter[str] = Counter()
    rounds_hist: Counter[int] = Counter()
    effective_rounds_hist: Counter[int] = Counter()
    acc_by_answer_round: dict[int, list[int]] = defaultdict(list)

    current: list[dict[str, Any]] = []

    def finalize(cur: list[dict[str, Any]]) -> None:
        nonlocal samples, answered, correct
        if not cur:
            return
        samples += 1

        # Find first valid answer (includes forced answer).
        ans_letter: Optional[str] = None
        ans_round: Optional[int] = None
        gt_idx: int = -1
        num_choices: int = 0
        for obj in cur:
            raw = str(obj.get("raw_output") or "")
            answer = _extract_tag(raw, _ANSWER_RE)
            if not answer:
                continue
            choices = obj.get("choices") or []
            num_choices = len(choices) if isinstance(choices, list) else 0
            letter = _normalize_answer_letter(answer, num_choices)
            if letter is None:
                continue
            ans_letter = letter
            ans_round = int(obj.get("round_idx") or 0)
            gt_idx = _safe_int(obj.get("ground_truth_idx"), -1)
            break

        # Effective rounds: count assistant turns that requested at least one NEW valid frame.
        eff = 0
        for obj in cur:
            raw = str(obj.get("raw_output") or "")
            if _extract_tag(raw, _ANSWER_RE):
                continue
            L = _extract_frame_count(str(obj.get("user_text") or ""))
            seen = set(int(i) for i in (obj.get("seen_frames") or []))

            # Prefer already-mapped requests when available (handles candidate ID action space).
            mapped = obj.get("requested_mapped_frames")
            if isinstance(mapped, list) and mapped:
                req = [int(i) for i in mapped]
            else:
                frames_text = _extract_tag(raw, _SELECT_RE)
                if frames_text is None:
                    continue
                req = _parse_frame_indices(frames_text)
            if not req:
                continue
            valid = [i for i in req if 0 <= i < L and i not in seen]
            if valid:
                eff += 1
        effective_rounds_hist[eff] += 1

        # Rounds used: if answered, cap by max-rounds; else use last observed round.
        if ans_round is not None:
            answered += 1
            used = min(int(ans_round), int(max_rounds))
            pred_idx = ord(ans_letter) - ord("A")
            is_correct = int(gt_idx >= 0 and pred_idx == gt_idx)
            correct += is_correct
            acc_by_answer_round[used].append(is_correct)
        else:
            used = int(cur[-1].get("round_idx") or 0)
        rounds_hist[used] += 1

        if ans_round is not None:
            return

        # No answer => infer termination reason from the last output.
        last = cur[-1]
        invalid_reason = str(last.get("invalid_reason") or "").strip()
        if invalid_reason:
            term_reasons[invalid_reason] += 1
            return
        raw = str(last.get("raw_output") or "")
        if _extract_tag(raw, _THINK_RE) is None:
            term_reasons["missing_think"] += 1
            return
        if _extract_tag(raw, _SELECT_RE) is None:
            term_reasons["missing_frames_tag"] += 1
            return
        frames_text = _extract_tag(raw, _SELECT_RE) or ""
        L = _extract_frame_count(str(last.get("user_text") or ""))
        seen = set(int(i) for i in (last.get("seen_frames") or []))
        mapped = last.get("requested_mapped_frames")
        if isinstance(mapped, list) and mapped:
            req = [int(i) for i in mapped]
        else:
            req = _parse_frame_indices(frames_text)
        valid = [i for i in req if 0 <= i < L and i not in seen]
        if not valid:
            term_reasons["invalid_frames"] += 1
        else:
            term_reasons["no_answer_but_valid_frames"] += 1

    for current in _group_prompt_rows(rows):
        finalize(current)

    out: dict[str, Any] = {
        "samples": samples,
        "answered": answered,
        "answered_acc": (correct / answered) if answered else 0.0,
        "overall_acc": (correct / samples) if samples else 0.0,
        "termination_reasons": _counter_json(term_reasons),
        "rounds_hist": _counter_json(rounds_hist),
        "effective_rounds_hist": _counter_json(effective_rounds_hist),
        "acc_by_answer_round": {
            str(k): (sum(v) / len(v) if v else 0.0) for k, v in sorted(acc_by_answer_round.items())
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-glob",
        required=True,
        help="Glob for prompts.jsonl files (e.g., '/path/to/run/shard_*/prompts.jsonl')",
    )
    ap.add_argument("--max-rounds", type=int, default=4, help="Round budget used in the run.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.log_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.log_glob}")

    lines = iter_log_lines(paths)
    if not lines:
        raise SystemExit("No JSONL lines found.")

    out = analyze_prompt_rows(lines, max_rounds=args.max_rounds)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
