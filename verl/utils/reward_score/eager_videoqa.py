# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""EAGER-style reward for REVISE video QA (approximate)."""

from __future__ import annotations

import re
import string
from typing import Any


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SUMMARIZE_RE = re.compile(r"<summarize>(.*?)</summarize>", re.DOTALL | re.IGNORECASE)


def _normalize(text: str) -> str:
    text = text.lower()
    text = " ".join(text.split())
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def _extract_tag(text: str, pattern: re.Pattern[str]) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _parse_answer(answer_text: str, choices: list[str]) -> int | None:
    # Try letter first
    letter_match = re.search(r"\b([A-E])\b", answer_text, re.IGNORECASE)
    if letter_match:
        return ord(letter_match.group(1).upper()) - ord("A")

    norm_answer = _normalize(answer_text)
    for idx, choice in enumerate(choices):
        if _normalize(choice) == norm_answer:
            return idx
    for idx, choice in enumerate(choices):
        if _normalize(choice) in norm_answer:
            return idx
    return None


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs,
):
    """Compute an approximate EAGER-style reward.

    This implementation prioritizes format validity, answer correctness,
    and summary sufficiency. Confidence gain and early-stop bonuses can be
    provided via extra_info['revise'].
    """

    extra_info = extra_info or {}
    revise_info = extra_info.get("revise", {})

    choices = ground_truth.get("choices", [])
    answer_idx = ground_truth.get("answer_idx")
    answer_text = ground_truth.get("answer_text")

    pred_answer = _extract_tag(solution_str, _ANSWER_RE)
    summary_text = _extract_tag(solution_str, _SUMMARIZE_RE)

    format_valid = pred_answer is not None
    pred_idx = _parse_answer(pred_answer, choices) if pred_answer else None

    answer_correct = False
    if pred_idx is not None and answer_idx is not None:
        try:
            answer_correct = pred_idx == int(answer_idx)
        except (ValueError, TypeError):
            answer_correct = False
    elif pred_answer and answer_text:
        answer_correct = _normalize(pred_answer) == _normalize(str(answer_text))

    summary_only_correct = revise_info.get("summary_only_correct")
    if summary_only_correct is None and summary_text and answer_text:
        summary_only_correct = _normalize(str(answer_text)) in _normalize(summary_text)
    summary_only_correct = bool(summary_only_correct)

    num_rounds = revise_info.get("num_rounds") or extra_info.get("num_turns")
    stop_round_threshold = revise_info.get("stop_round_threshold", 2)

    # Weights (defaults are conservative)
    weights = revise_info.get("reward_weights", {})
    lambda_conf = float(weights.get("conf", 0.0))
    lambda_sum = float(weights.get("sum", 1.0))
    lambda_stop = float(weights.get("stop", 0.5))
    format_bonus = float(weights.get("format", 0.1))
    answer_bonus = float(weights.get("answer", 1.0))

    r_conf = float(revise_info.get("conf_gain", 0.0))
    r_sum = 1.0 if summary_only_correct else 0.0
    r_stop = 0.0
    if answer_correct and num_rounds is not None and int(num_rounds) <= int(stop_round_threshold):
        r_stop = 1.0
    r_format = 1.0 if format_valid else 0.0

    score = lambda_conf * r_conf + lambda_sum * r_sum + lambda_stop * r_stop + format_bonus * r_format
    if answer_correct:
        score += answer_bonus

    return {
        "score": score,
        "answer_correct": float(answer_correct),
        "summary_only_correct": float(summary_only_correct),
        "format_valid": float(format_valid),
        "r_conf": r_conf,
        "r_sum": r_sum,
        "r_stop": r_stop,
        "r_format": r_format,
    }
