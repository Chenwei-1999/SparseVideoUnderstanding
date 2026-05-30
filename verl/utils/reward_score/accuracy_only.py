from __future__ import annotations

import re
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer(solution_str: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(solution_str))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _parse_answer_idx(answer_text: str, num_choices: int) -> int | None:
    if not answer_text:
        return None
    match = re.search(r"\b([A-E])\b", answer_text, re.IGNORECASE)
    if not match:
        match = re.search(r"([A-E])", answer_text, re.IGNORECASE)
    if not match:
        return None
    idx = ord(match.group(1).upper()) - ord("A")
    if 0 <= idx < max(1, num_choices):
        return idx
    return None


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Reward = final accuracy only (1.0 if correct else 0.0)."""

    choices = ground_truth.get("choices") or []
    answer_idx = ground_truth.get("answer_idx")
    if answer_idx is None:
        return {"score": 0.0, "answer_correct": 0.0, "format_valid": 0.0}

    answer_text = _extract_answer(solution_str)
    pred_idx = _parse_answer_idx(answer_text or "", len(choices))
    format_valid = float(pred_idx is not None)
    answer_correct = float(pred_idx is not None and int(pred_idx) == int(answer_idx))
    return {
        "score": answer_correct,
        "answer_correct": answer_correct,
        "format_valid": format_valid,
    }

