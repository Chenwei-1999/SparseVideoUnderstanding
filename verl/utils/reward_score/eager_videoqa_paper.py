"""EAGER reward (paper-faithful) for REVISE-style video QA.

This implementation matches the paper's dense, annotation-free formulation:

Per-step reward:
    r_t = λ1 * r_conf_t + λ2 * r_sum_t + λ3 * r_stop_t + r_format_t

Episode return:
    R(H) = Σ_{t=1..τ} γ^{t-1} r_t

Signals are expected to be produced during rollout by the REVISE agent loop and passed
via `extra_info['revise']`:
  - margins: list[float] of m_t at each decision state
  - actions: list[str] in {"select","answer"} aligned with margins
  - summary_only_correct: 0/1 (computed at answer time)
  - format_by_round: list[float] (0/1) aligned with actions

No frame-level annotations are used; only answer labels (y*) and model scores.
"""

from __future__ import annotations

import re
from typing import Any, Optional


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = list(_ANSWER_RE.finditer(text))
    if not m:
        return None
    raw = m[-1].group(1).strip().upper()
    if not raw:
        return None
    match = re.search(r"\b([A-E])\b", raw)
    if match:
        return match.group(1).upper()
    match = re.search(r"([A-E])", raw)
    if match:
        return match.group(1).upper()
    return None


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    *,
    lambda_conf: float = 1.0,
    lambda_sum: float = 1.0,
    lambda_stop: float = 1.0,
    gamma: float = 0.99,
    format_reward: float = 0.05,
    stop_round_threshold: int = 2,
    # Optional shaping: add extra early-stop bonus within the threshold.
    # When enabled, r_stop becomes: 1 + β * (stop_round_threshold - num_rounds) for correct early stops.
    stop_bonus_beta: float = 0.0,
    **_: Any,
) -> dict[str, float]:
    extra_info = extra_info or {}
    revise = extra_info.get("revise", {}) if isinstance(extra_info, dict) else {}

    # Always return a stable key-set to keep Ray collation happy.
    out: dict[str, float] = {
        "score": 0.0,
        "return": 0.0,
        "answer_correct": 0.0,
        "summary_only_correct": 0.0,
        "stop_early": 0.0,
        "illegal_action": 0.0,
        "tau": 0.0,
        "sum_r_conf": 0.0,
        "sum_r_sum": 0.0,
        "sum_r_stop": 0.0,
        "num_format_ok": 0.0,
    }

    # Required GT fields.
    answer_idx = ground_truth.get("answer_idx") if isinstance(ground_truth, dict) else None
    if answer_idx is None:
        return out

    num_choices = len(ground_truth.get("choices") or [])
    if num_choices <= 0:
        num_choices = 5

    try:
        correct_letter = chr(ord("A") + int(answer_idx))
    except Exception:
        correct_letter = None

    # Predicted answer (from revise extra_info preferred).
    pred_letter = revise.get("answer")
    if isinstance(pred_letter, str):
        pred_letter = pred_letter.strip().upper() or None
    else:
        pred_letter = None
    if pred_letter is None:
        pred_letter = _extract_answer_letter(solution_str or "")

    allowed = {chr(ord("A") + i) for i in range(num_choices)}
    if pred_letter not in allowed:
        pred_letter = None

    answer_correct = float(pred_letter == correct_letter) if (pred_letter and correct_letter) else 0.0
    out["answer_correct"] = float(answer_correct)

    num_rounds = revise.get("num_rounds")
    try:
        num_rounds_i = int(num_rounds) if num_rounds is not None else None
    except Exception:
        num_rounds_i = None

    stop_early = float(
        answer_correct > 0.0
        and num_rounds_i is not None
        and int(num_rounds_i) <= int(stop_round_threshold)
    )
    out["stop_early"] = float(stop_early)

    # EAGER signals from agent loop.
    actions = revise.get("actions") or []
    margins = revise.get("margins") or []
    format_by_round = revise.get("format_by_round") or []

    # Summary-only correctness (computed in the agent loop via model scores).
    try:
        summary_only_correct = float(revise.get("summary_only_correct") or 0.0)
    except Exception:
        summary_only_correct = 0.0
    out["summary_only_correct"] = float(summary_only_correct)

    # If the rollout terminated due to an illegal action, keep reward near zero.
    illegal_action = int(revise.get("illegal_action") or 0)
    out["illegal_action"] = float(illegal_action)
    if illegal_action:
        return out

    # Defensive alignment: only compute per-step reward when lists look sane.
    if not isinstance(actions, list) or not isinstance(margins, list):
        return out

    tau = min(len(actions), len(margins))
    if tau <= 0:
        return out

    total_conf = 0.0
    total_sum = 0.0
    total_stop = 0.0
    total_format = 0.0

    returns = 0.0
    discount = 1.0
    for t in range(tau):
        act = str(actions[t]).lower()

        r_conf = 0.0
        if act == "select":
            # r_conf_t = [m_{t+1} - m_t]_+ (index-shifted; aligns with paper "applies on Select").
            if t + 1 < len(margins):
                try:
                    r_conf = max(float(margins[t + 1]) - float(margins[t]), 0.0)
                except Exception:
                    r_conf = 0.0

        r_sum = 0.0
        r_stop = 0.0
        if act == "answer":
            r_sum = 1.0 if summary_only_correct > 0.0 else 0.0
            if stop_early > 0.0:
                delta = max(int(stop_round_threshold) - int(num_rounds_i or 0), 0)
                r_stop = 1.0 + float(stop_bonus_beta) * float(delta)

        r_format = 0.0
        if format_reward:
            try:
                ok = float(format_by_round[t]) if t < len(format_by_round) else 0.0
            except Exception:
                ok = 0.0
            r_format = float(format_reward) if ok > 0.0 else 0.0

        r_t = float(lambda_conf) * r_conf + float(lambda_sum) * r_sum + float(lambda_stop) * r_stop + r_format
        returns += discount * r_t
        discount *= float(gamma)

        total_conf += r_conf
        total_sum += r_sum
        total_stop += r_stop
        total_format += (1.0 if r_format > 0.0 else 0.0)

    out.update(
        {
            "score": float(returns),
            "return": float(returns),
            "tau": float(tau),
            "sum_r_conf": float(total_conf),
            "sum_r_sum": float(total_sum),
            "sum_r_stop": float(total_stop),
            "num_format_ok": float(total_format),
        }
    )
    return out
