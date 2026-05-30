# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""Safer EAGER-style reward for REVISE video QA.

This version avoids rewarding summary "answer_text" substring matches (which can be trivially
hacked by listing all options). Instead, it rewards the hypothesis letter in the summary's
`H:` field when it matches the ground-truth option letter.
"""

from __future__ import annotations

import re
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SELECT_RE = re.compile(r"<select>(.*?)</select>", re.DOTALL | re.IGNORECASE)
_SUMMARIZE_RE = re.compile(r"<summarize>(.*?)</summarize>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_HYP_RE = re.compile(r"\bH\s*:\s*([A-E])\b", re.IGNORECASE)

# Paper protocol: <think> is REQUIRED first. Answer = think+answer only (no repeated summarize).
_STRICT_ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>\s*([A-E])\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_STRICT_SELECT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<summarize>.*?</summarize>\s*<select>\s*\d+(?:\s*,\s*\d+)*\s*</select>\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _is_placeholder(text: str) -> bool:
    t = _collapse_ws(text).lower()
    if not t:
        return True
    if t in {"...", "…", "none", "n/a", "na", "null"}:
        return True
    if t in {"unknown", "unsure", "uncertain"}:
        return True
    if re.fullmatch(r"[.·•…]+", t):
        return True
    # Too short: often a placeholder like "ok", "idk", etc.
    alnum = re.findall(r"[a-z0-9]+", t)
    if len(alnum) <= 1 and len(t) <= 6:
        return True
    return False


def _text_quality(text: str, *, min_words: int) -> float:
    if text is None:
        return 0.0
    if _is_placeholder(text):
        return 0.0
    words = re.findall(r"[A-Za-z0-9]+", str(text))
    if len(words) < min_words:
        return 0.5
    return 1.0


def _summary_quality(summary_text: str, num_choices: int) -> float:
    """Return 1.0 for well-formed, non-placeholder O/H/R/P/U summary; else 0.0."""
    if summary_text is None:
        return 0.0
    summary = _collapse_ws(summary_text)
    if _is_placeholder(summary):
        return 0.0

    # Require O/H/R/P/U labels.
    required = ["O", "H", "R", "P", "U"]
    for key in required:
        if re.search(rf"\b{key}\s*:", summary, re.IGNORECASE) is None:
            return 0.0

    # Extract H field and ensure exactly one option letter appears.
    h_match = re.search(r"\bH\s*:\s*([^;]+)", summary, re.IGNORECASE)
    if not h_match:
        return 0.0
    h_field = h_match.group(1)
    letters = re.findall(r"\b([A-E])\b", h_field, re.IGNORECASE)
    unique_letters = {letter.upper() for letter in letters}
    if len(unique_letters) != 1:
        return 0.0
    if _normalize_answer_letter(next(iter(unique_letters)), num_choices) is None:
        return 0.0

    # Extract P field and require at least one frame index.
    p_match = re.search(r"\bP\s*:\s*([^;]+)", summary, re.IGNORECASE)
    if not p_match:
        return 0.0
    if not re.findall(r"\d+", p_match.group(1)):
        return 0.0

    # Avoid placeholder/empty content in O/R/U.
    for key in ["O", "R", "U"]:
        m = re.search(rf"\b{key}\s*:\s*([^;]+)", summary, re.IGNORECASE)
        if not m or _is_placeholder(m.group(1)):
            return 0.0
    return 1.0


def _extract_tag(text: str, pattern: re.Pattern[str]) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _normalize_answer_letter(answer_text: str, num_choices: int) -> str | None:
    allowed = {chr(ord("A") + i) for i in range(max(0, num_choices))}
    if not allowed:
        allowed = {"A", "B", "C", "D", "E"}

    candidate = answer_text.strip().upper()
    if candidate in allowed:
        return candidate

    match = re.search(r"\b([A-E])\b", candidate)
    if match:
        letter = match.group(1).upper()
        return letter if letter in allowed else None

    match = re.search(r"([A-E])", candidate)
    if match:
        letter = match.group(1).upper()
        return letter if letter in allowed else None
    return None


def _extract_summary_hyp_letter(summary_text: str, num_choices: int) -> str | None:
    if not summary_text:
        return None
    match = _HYP_RE.search(summary_text)
    if match:
        return _normalize_answer_letter(match.group(1), num_choices)
    # Fall back to any single letter (avoid rewarding lists like "A/B/C/D/E").
    match = re.search(r"\b([A-E])\b", summary_text, re.IGNORECASE)
    if match:
        return _normalize_answer_letter(match.group(1), num_choices)
    return None


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    reward_weights: dict[str, float] | None = None,
    stop_round_threshold: int = 2,
    **kwargs: Any,
) -> dict[str, float]:
    """Compute a conservative EAGER-style reward.

    Reward components:
    - answer_correct: 1 if final <answer> matches ground-truth option index.
    - format_valid: 1 if <answer> is present and parsable.
    - format_strict: 1 if the final assistant message matches exactly one required format (no mixed tags).
    - think_quality: 0/0.5/1 based on non-placeholder <think> content.
    - summary_quality: 1 if summary has meaningful O/H/R/P/U structure (only applied when multi-round).
    - summary_hyp_correct: 1 if <summarize> contains a single-letter hypothesis matching the label.
    - stop_early: 1 if answer is correct and num_rounds <= stop_round_threshold.

    The summary bonus is based on `H:` letter to avoid substring hacks.
    """

    extra_info = extra_info or {}
    revise_info = extra_info.get("revise", {}) if isinstance(extra_info, dict) else {}

    choices = ground_truth.get("choices") or []
    answer_idx = ground_truth.get("answer_idx")
    if answer_idx is None:
        return {
            "score": 0.0,
            "answer_correct": 0.0,
            "format_valid": 0.0,
            "summary_hyp_correct": 0.0,
            "stop_early": 0.0,
        }

    num_choices = len(choices) if isinstance(choices, list) else 0
    try:
        idx_int = int(answer_idx)
    except (ValueError, TypeError):
        return {
            "score": 0.0,
            "answer_correct": 0.0,
            "format_valid": 0.0,
            "summary_hyp_correct": 0.0,
            "stop_early": 0.0,
        }
    if idx_int < 0 or idx_int >= max(num_choices, 5):
        return {
            "score": 0.0,
            "answer_correct": 0.0,
            "format_valid": 0.0,
            "summary_hyp_correct": 0.0,
            "stop_early": 0.0,
        }
    correct_letter = chr(ord("A") + idx_int)

    pred_answer_text = _extract_tag(solution_str, _ANSWER_RE)
    pred_letter = _normalize_answer_letter(pred_answer_text or "", num_choices)
    format_valid = float(pred_letter is not None)
    answer_correct = float(pred_letter == correct_letter) if pred_letter is not None else 0.0

    summary_text = revise_info.get("summary") or _extract_tag(solution_str, _SUMMARIZE_RE) or ""
    hyp_letter = _extract_summary_hyp_letter(str(summary_text), num_choices)
    summary_hyp_correct = float(hyp_letter == correct_letter) if hyp_letter is not None else 0.0

    num_rounds = revise_info.get("num_rounds")
    if num_rounds is None:
        num_rounds = extra_info.get("num_turns")
    stop_early = float(answer_correct > 0.0 and num_rounds is not None and int(num_rounds) <= int(stop_round_threshold))

    # Strict format and quality checks should run on the final assistant response only.
    final_response = str(revise_info.get("last_response") or solution_str)
    has_answer = _extract_tag(final_response, _ANSWER_RE) is not None
    if has_answer:
        strict_match = _STRICT_ANSWER_RE.match(final_response)
        if strict_match:
            strict_letter = _normalize_answer_letter(strict_match.group(1), num_choices)
            format_strict = float(strict_letter is not None)
        else:
            format_strict = 0.0
    else:
        format_strict = float(_STRICT_SELECT_RE.match(final_response) is not None)

    think_text = _extract_tag(final_response, _THINK_RE) or ""
    think_quality = float(_text_quality(think_text, min_words=4))

    summary_quality = 1.0
    if num_rounds is not None and int(num_rounds) > 1:
        summary_quality = float(_summary_quality(str(summary_text), num_choices))

    # Default weights: keep close to accuracy-only while adding small shaping terms.
    weights = reward_weights or {}
    w_answer = float(weights.get("answer", 1.0))
    w_format = float(weights.get("format", 0.1))
    w_sum = float(weights.get("sum", 0.2))
    w_stop = float(weights.get("stop", 0.2))
    w_strict = float(weights.get("strict", 0.0))
    w_think = float(weights.get("think", 0.0))
    w_summary = float(weights.get("summary", 0.0))

    score = (
        w_answer * answer_correct
        + w_format * format_valid
        + w_sum * summary_hyp_correct
        + w_stop * stop_early
        + w_strict * format_strict
        + w_think * think_quality
        + w_summary * summary_quality
    )

    return {
        "score": float(score),
        "answer_correct": float(answer_correct),
        "format_valid": float(format_valid),
        "format_strict": float(format_strict),
        "think_quality": float(think_quality),
        "summary_quality": float(summary_quality),
        "summary_hyp_correct": float(summary_hyp_correct),
        "stop_early": float(stop_early),
        "pred_letter": float(ord(pred_letter) - ord("A")) if pred_letter is not None else -1.0,
        "hyp_letter": float(ord(hyp_letter) - ord("A")) if hyp_letter is not None else -1.0,
        "num_rounds": float(num_rounds) if num_rounds is not None else -1.0,
    }
