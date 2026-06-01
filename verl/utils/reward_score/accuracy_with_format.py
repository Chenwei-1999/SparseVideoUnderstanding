from __future__ import annotations

import re
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SELECT_RE = re.compile(r"<select>(.*?)</select>", re.DOTALL | re.IGNORECASE)
_SUMMARIZE_RE = re.compile(r"<summarize>(.*?)</summarize>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Paper protocol: every response begins with <think>. Select = think+summarize+select;
# Answer = think+answer only (the last committed summarize is reused, not repeated).
_STRICT_SELECT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<summarize>.*?</summarize>\s*<select>\s*\d+(?:\s*,\s*\d+)*\s*</select>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_STRICT_ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>\s*([A-E])\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _is_placeholder(text: str) -> bool:
    t = _collapse_ws(text).lower()
    if not t:
        return True
    # Treat ellipses anywhere as placeholders.
    if "..." in t or "…" in t:
        return True
    if t in {"...", "…", "none", "n/a", "na", "null", "unknown", "unsure", "uncertain"}:
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


def _extract_tag(text: str, pattern: re.Pattern[str]) -> str | None:
    matches = list(pattern.finditer(text or ""))
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


def _summary_quality(summary_text: str, num_choices: int) -> float:
    """Return a quality score for a well-formed, non-placeholder O/H/R/P/U summary.

    We reward informative, non-placeholder content (O/H/R/U) and a concrete, natural-language
    description of previously seen frames (P). The score is only used as a conditional bonus
    when the final answer is correct (see `compute_score`).
    """
    if summary_text is None:
        return 0.0
    summary = _collapse_ws(summary_text)
    if _is_placeholder(summary):
        return 0.0

    # Require P/O/H/U/R labels in this order.
    order = ["P", "O", "H", "U", "R"]
    for key in order:
        if re.search(rf"\b{key}\s*:", summary, re.IGNORECASE) is None:
            return 0.0
    positions = []
    for key in order:
        m = re.search(rf"\b{key}\s*:\s*", summary, re.IGNORECASE)
        if m is None:
            return 0.0
        positions.append(m.start())
    if not all(a < b for a, b in zip(positions, positions[1:], strict=False)):
        return 0.0

    def _field(key: str) -> str | None:
        m = re.search(
            rf"\b{key}\s*:\s*(.*?)(?=\b(?:P|O|H|U|R)\s*:|$)",
            summary,
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        return _collapse_ws(m.group(1))

    p_text = _field("P")
    o_text = _field("O")
    h_text = _field("H")
    u_text = _field("U")
    r_text = _field("R")
    if any(v is None for v in [o_text, h_text, r_text, p_text, u_text]):
        return 0.0

    # Reward informative, non-placeholder content for O/H/U/R.
    o_q = _text_quality(o_text, min_words=4)
    h_q = _text_quality(h_text, min_words=4)
    u_q = _text_quality(u_text, min_words=4)
    r_q = _text_quality(r_text, min_words=4)

    # P: natural-language recap of previously seen frames (avoid Python list formatting like "[4, 8, 12]").
    p_q = _text_quality(p_text, min_words=4)
    if "[" in p_text or "]" in p_text:
        p_q = 0.0

    return float((o_q + h_q + r_q + p_q + u_q) / 5.0)


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    reward_weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Reward = accuracy + a small conditional formatting bonus.

    The formatting bonus/penalties are only applied when the final answer is correct, to avoid the
    model trading off accuracy for protocol/format hacking.
    """

    choices = ground_truth.get("choices") or []
    answer_idx = ground_truth.get("answer_idx")
    if answer_idx is None:
        return {
            "score": 0.0,
            "answer_correct": 0.0,
            "format_valid": 0.0,
            "format_strict": 0.0,
            "think_quality": 0.0,
            "summary_quality": 0.0,
            "summary_present": 0.0,
        }

    num_choices = len(choices) if isinstance(choices, list) else 0
    correct_letter = chr(ord("A") + int(answer_idx))

    extra_info = extra_info or {}
    revise_info = extra_info.get("revise", {}) if isinstance(extra_info, dict) else {}
    final_response = str(revise_info.get("last_response") or solution_str)

    num_rounds = int(revise_info.get("num_rounds") or 0)
    effective_rounds = int(revise_info.get("effective_rounds") or 0)

    invalid_action = int(
        revise_info.get("illegal_action")
        or revise_info.get("invalid_action")
        or revise_info.get("terminated_invalid_action")
        or 0
    )

    answer_text = _extract_tag(final_response, _ANSWER_RE)
    pred_letter = _normalize_answer_letter(answer_text or "", num_choices)
    format_valid = float(pred_letter is not None)
    answer_correct = float(pred_letter == correct_letter) if pred_letter is not None else 0.0

    has_answer = answer_text is not None
    if has_answer:
        format_strict = float(_STRICT_ANSWER_RE.match(final_response) is not None)
    else:
        format_strict = float(_STRICT_SELECT_RE.match(final_response) is not None)

    # The REVISE protocol REQUIRES a <think> trace first; a missing <think> scores zero.
    think_text = _extract_tag(final_response, _THINK_RE)
    think_quality = float(_text_quality(think_text, min_words=4)) if think_text is not None else 0.0

    summary_text = str(revise_info.get("summary") or _extract_tag(final_response, _SUMMARIZE_RE) or "")
    summary_present = float(bool(summary_text))
    summary_quality = float(_summary_quality(summary_text, num_choices)) if summary_text else 0.0

    weights = reward_weights or {}
    w_answer = float(weights.get("answer", 1.0))
    w_format = float(weights.get("format", 0.05))
    w_think = float(weights.get("think", 0.02))
    w_summary = float(weights.get("summary", 0.05))
    w_invalid_penalty = float(weights.get("invalid_penalty", 0.0))
    w_all_seen_penalty = float(weights.get("frames_all_seen_penalty", 0.0))
    w_illegal_action_penalty = float(weights.get("illegal_action_penalty", 0.0))
    w_effective_round_bonus = float(weights.get("effective_round_bonus", 0.0))
    w_incorrect_answer_penalty = float(weights.get("incorrect_answer_penalty", 0.0))

    # Immediate termination with a strong negative reward for illegal actions.
    if invalid_action > 0 and w_illegal_action_penalty > 0:
        score = -abs(w_illegal_action_penalty) * float(invalid_action)
        return {
            "score": float(score),
            "answer_correct": float(answer_correct),
            "format_valid": float(format_valid),
            "format_strict": float(format_strict),
            "think_quality": float(think_quality),
            "summary_present": float(summary_present),
            "summary_quality": float(summary_quality),
            "invalid_outputs": float(int(revise_info.get("invalid_outputs") or 0)),
            "frames_all_seen": float(int(revise_info.get("frames_all_seen") or 0)),
            "illegal_action": float(invalid_action),
            "num_rounds": float(num_rounds),
            "effective_rounds": float(effective_rounds),
            "pred_letter": float(ord(pred_letter) - ord("A")) if pred_letter is not None else -1.0,
        }

    bonus = (w_format * format_strict) + (w_think * think_quality) + (w_summary * summary_quality)

    invalid_outputs = int(revise_info.get("invalid_outputs") or 0)
    frames_all_seen = int(revise_info.get("frames_all_seen") or 0)
    penalty = (w_invalid_penalty * float(invalid_outputs)) + (w_all_seen_penalty * float(frames_all_seen))

    round_bonus = w_effective_round_bonus * float(effective_rounds)
    incorrect_penalty = float(abs(w_incorrect_answer_penalty)) if (format_valid > 0 and answer_correct <= 0) else 0.0
    score = (
        round_bonus
        + (w_answer * answer_correct)
        + (answer_correct * bonus)
        - (answer_correct * penalty)
        - incorrect_penalty
    )

    return {
        "score": float(score),
        "answer_correct": float(answer_correct),
        "format_valid": float(format_valid),
        "format_strict": float(format_strict),
        "think_quality": float(think_quality),
        "summary_present": float(summary_present),
        "summary_quality": float(summary_quality),
        "invalid_outputs": float(invalid_outputs),
        "frames_all_seen": float(frames_all_seen),
        "illegal_action": float(invalid_action),
        "num_rounds": float(num_rounds),
        "effective_rounds": float(effective_rounds),
        "pred_letter": float(ord(pred_letter) - ord("A")) if pred_letter is not None else -1.0,
    }
