"""Shared REVISE controller policy.

This module is intentionally backend-free. Plug-and-play evaluation and RL
training both call into it after a model turn, so prompt/action semantics stay
identical while transports, rewards, and training remain separate concerns.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional

from revise.pnp.prompts import SYSTEM_PROMPT
from revise.pnp.utils import (
    FORCE_ANSWER_INSTRUCTIONS_POHR,
    OPTION_LABELS,
    contains_banned_example,
    dedupe_preserve_order,
    format_intervals,
    indices_to_intervals,
    is_placeholder,
    normalize_answer_letter,
    parse_int_list,
    parse_strict_revise_action,
    retry_feedback_text,
    should_retry_revise_invalid_output,
    summary_has_ohrpu,
    unseen_intervals,
)


@dataclass(frozen=True)
class ReviseActionDecision:
    """Validated model action for one REVISE turn."""

    kind: str
    reason: Optional[str] = None
    think: Optional[str] = None
    summary: Optional[str] = None
    answer_letter: Optional[str] = None
    select_text: Optional[str] = None
    requested_raw_frames: tuple[int, ...] = field(default_factory=tuple)
    requested_frames: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def invalid(cls, reason: str) -> ReviseActionDecision:
        return cls(kind="invalid", reason=reason)


@dataclass(frozen=True)
class InvalidReviseActionResolution:
    """Shared control decision after an invalid REVISE action."""

    kind: str
    reason: str
    terminated_invalid_action: bool = False


def final_answer_instruction() -> str:
    """Canonical final-round instruction shared by PnP and RL loops."""

    return "This is the final round. You MUST answer now using <think>...</think> then <answer>LETTER</answer>."


def resolve_invalid_revise_action(
    reason: str,
    *,
    retry_idx: Optional[int] = None,
    max_retries_per_round: Optional[int] = None,
    retries_left: Optional[int] = None,
    retryable: bool = True,
    strict_actions: bool = True,
    clear_frame_plan: bool = False,
) -> InvalidReviseActionResolution:
    """Resolve retry/termination/fallback after a malformed REVISE turn.

    PnP and RL have different transports, but Figure-3 protocol violations
    should move through the same control states: retry while budget remains,
    terminate in strict paper settings, and only then enter explicit non-paper
    fallback behavior.
    """

    if should_retry_revise_invalid_output(
        reason,
        retry_idx=retry_idx,
        max_retries_per_round=max_retries_per_round,
        retries_left=retries_left,
        retryable=retryable,
    ):
        return InvalidReviseActionResolution(kind="retry", reason=reason)
    if bool(strict_actions):
        return InvalidReviseActionResolution(kind="terminate", reason=reason, terminated_invalid_action=True)
    if bool(clear_frame_plan):
        return InvalidReviseActionResolution(kind="clear_frame_plan", reason=reason)
    return InvalidReviseActionResolution(kind="fallback", reason=reason)


def format_revise_system_prompt(max_frames_per_round: int, *, template: str = SYSTEM_PROMPT) -> str:
    """Format the single canonical REVISE system prompt."""

    return template.format(max_frames_per_round=int(max_frames_per_round))


def _default_summary_valid(summary: Optional[str], seen_count: int) -> bool:
    _ = seen_count
    return (
        summary is not None
        and not is_placeholder(summary)
        and not contains_banned_example(summary)
        and summary_has_ohrpu(summary)
    )


def _default_summary_stale(summary: Optional[str], seen_count: int) -> bool:
    _ = summary, seen_count
    return False


def _default_has_range_syntax(frames_text: str) -> bool:
    if not frames_text:
        return False
    return bool(re.search(r"\d+\s*[-–—]\s*\d+", frames_text))


def _is_strict_answer_body(answer_text: str, num_choices: int) -> bool:
    labels = OPTION_LABELS[: max(0, int(num_choices or 0))] or "ABCDE"
    return bool(re.fullmatch(rf"\s*[{re.escape(labels)}]\s*", str(answer_text or "").upper()))


def _is_strict_select_body(select_text: str) -> bool:
    return bool(re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", str(select_text or "")))


def decide_revise_action(
    raw_output: str,
    *,
    num_choices: int,
    seen_frames: Sequence[int],
    frame_count: int,
    max_frames_per_round: int,
    round_idx: int = 1,
    max_rounds: Optional[int] = None,
    answer_only_final_round: bool = False,
    require_answer: bool = False,
    min_select_rounds: int = 0,
    select_rounds_so_far: int = 0,
    candidate_frames: Optional[Sequence[int]] = None,
    use_candidate_frame_ids: bool = False,
    require_candidate_frames: bool = False,
    parse_select_frames: Callable[[str], list[int]] = parse_int_list,
    normalize_answer: Optional[Callable[[str], Optional[str]]] = None,
    is_select_summary_valid: Callable[[Optional[str], int], bool] = _default_summary_valid,
    is_summary_stale: Callable[[Optional[str], int], bool] = _default_summary_stale,
    select_has_range_syntax: Callable[[str], bool] = _default_has_range_syntax,
) -> ReviseActionDecision:
    """Parse and validate a Figure-3 REVISE action.

    Select rounds must be ``<think><summarize><select>`` and answer rounds must
    be ``<think><answer>``. The function returns either a validated ``answer``,
    a validated ``select`` frame plan, or an ``invalid`` reason that both PnP
    and RL handle with the same retry/termination policy.
    """

    parsed = parse_strict_revise_action(raw_output)
    if parsed is None:
        return ReviseActionDecision.invalid("invalid_paper_protocol")

    think = parsed["think"]
    summary = parsed["summary"]
    answer = parsed["answer"]
    select_text = parsed["select"]
    candidate_frames = [int(i) for i in (candidate_frames or [])]
    seen_set = {int(i) for i in seen_frames}
    normalize = normalize_answer or (lambda text: normalize_answer_letter(text, num_choices))

    def invalid(reason: str, requested_raw: Optional[Sequence[int]] = None) -> ReviseActionDecision:
        return ReviseActionDecision(
            kind="invalid",
            reason=reason,
            think=think,
            summary=summary,
            select_text=select_text,
            requested_raw_frames=tuple(int(i) for i in (requested_raw or [])),
        )

    if answer is not None:
        if not _is_strict_answer_body(answer, num_choices):
            return invalid("invalid_answer_letter")
        answer_letter = normalize(answer)
        if answer_letter is None:
            return invalid("invalid_answer_letter")
        if int(select_rounds_so_far or 0) < int(min_select_rounds or 0):
            return invalid("early_answer_disallowed")
        if bool(answer_only_final_round) and max_rounds is not None and int(round_idx) < int(max_rounds):
            return invalid("early_answer_disallowed")
        return ReviseActionDecision(
            kind="answer",
            think=think,
            answer_letter=answer_letter,
        )

    if select_text is None:
        return invalid("missing_frames_tag")

    if bool(require_answer):
        return invalid("final_select_disallowed")

    if not is_select_summary_valid(summary, len(seen_set)):
        return invalid("invalid_select_summary")
    if is_summary_stale(summary, len(seen_set)):
        return invalid("stale_select_summary")
    if not bool(use_candidate_frame_ids) and select_has_range_syntax(select_text):
        return invalid("frames_range_syntax")
    if not _is_strict_select_body(select_text):
        return invalid("invalid_frames")

    requested_raw = dedupe_preserve_order([int(i) for i in parse_select_frames(select_text)])
    if bool(use_candidate_frame_ids) and candidate_frames:
        mapped: list[int] = []
        for candidate_id in requested_raw:
            if not (1 <= int(candidate_id) <= len(candidate_frames)):
                return invalid("frames_out_of_range", requested_raw)
            mapped.append(int(candidate_frames[int(candidate_id) - 1]))
        requested = dedupe_preserve_order(mapped)
    else:
        if bool(require_candidate_frames) and candidate_frames:
            allowed = {int(i) for i in candidate_frames}
            if any(int(i) not in allowed for i in requested_raw):
                return invalid("frames_not_in_candidates", requested_raw)
        requested = requested_raw

    if not requested:
        return invalid("empty_frames", requested_raw)

    requested_seen = [int(i) for i in requested if int(i) in seen_set]
    if requested_seen:
        return invalid("frames_already_seen", requested_raw)

    requested_unseen = [int(i) for i in requested if int(i) not in seen_set]
    if not requested_unseen:
        return invalid("frames_all_seen", requested_raw)
    if len(requested_unseen) > int(max_frames_per_round):
        return invalid("too_many_frames", requested_raw)

    if any(not (0 <= int(i) < int(frame_count)) for i in requested_unseen):
        return invalid("frames_out_of_range", requested_raw)

    return ReviseActionDecision(
        kind="select",
        think=think,
        summary=summary,
        select_text=select_text,
        requested_raw_frames=tuple(requested_raw),
        requested_frames=tuple(requested_unseen),
    )


def build_retry_feedback(
    reason: str,
    *,
    force_answer: bool = False,
    max_frames_per_round: int = 0,
    frame_count: int = 0,
    seen_frames: Optional[Sequence[int]] = None,
    num_choices: int = 5,
    candidate_count: int = 0,
    candidate_frames: Optional[Sequence[int]] = None,
    force_instructions: str = FORCE_ANSWER_INSTRUCTIONS_POHR,
) -> str:
    """Build canonical retry text for an invalid REVISE action."""

    k = int(max_frames_per_round or 0)
    seen = [int(i) for i in (seen_frames or [])]
    labels = ", ".join(OPTION_LABELS[: max(0, int(num_choices or 0))] or "ABCDE")
    candidate_count = int(candidate_count or len(candidate_frames or []))
    unseen_text = format_intervals(unseen_intervals(int(frame_count or 0), seen))
    candidate_text = format_intervals(indices_to_intervals([int(i) for i in (candidate_frames or [])]))

    if reason == "missing_think":
        feedback = (
            "Invalid response: every response MUST begin with <think>...</think>, "
            "then either <summarize>...</summarize> plus <select>...</select> or <answer>LETTER</answer>."
        )
    elif reason == "invalid_paper_protocol":
        feedback = (
            "Invalid response: output exactly one paper protocol action. "
            "Select rounds must be <think>...</think> then <summarize>...</summarize> then "
            "<select>...</select>. Answer rounds must be <think>...</think> then <answer>LETTER</answer>. "
            "Do not mix <answer> with <select> or <summarize>."
        )
    elif reason == "invalid_answer_letter":
        feedback = (
            "Invalid response: <answer> must be exactly ONE option letter "
            f"({labels}). Do not output words or a sentence."
        )
    elif reason == "early_answer_disallowed":
        feedback = (
            "Invalid response: do NOT answer yet. Request more frames using "
            "<think>...</think> then <summarize>...</summarize> then <select>...</select>."
        )
    elif reason == "final_select_disallowed":
        feedback = (
            "Invalid response: this is the final round, so do NOT request more frames. "
            "Answer now using <think>...</think> then <answer>LETTER</answer>."
        )
    elif reason == "missing_frames_tag":
        feedback = (
            "Invalid response: missing <select> tag for requesting more frames. "
            "Remember: <select> must list NEW frame indices to view NEXT."
        )
    elif reason == "invalid_select_summary":
        feedback = (
            "Invalid response: when requesting more frames, include a meaningful <summarize> with P/O/H/U/R "
            "in that exact order and no placeholders such as 'none' or 'unknown'."
        )
    elif reason == "stale_select_summary":
        feedback = (
            "Invalid response: the <summarize> claims no frames/captions were seen, but evidence was shown. "
            "Rewrite <summarize> to reflect observed evidence so far, then request frames."
        )
    elif reason == "frames_range_syntax":
        feedback = (
            "Invalid response: <select> must be a comma-separated list of integers only "
            f"(NO ranges like '4-182'). Choose up to {k} NEW frames."
        )
    elif reason == "frames_out_of_range" and candidate_count > 0:
        feedback = (
            "Invalid response: when Candidate Frame IDs are provided, output IDs "
            f"in [1, {candidate_count}] inside <select>."
        )
    elif reason == "frames_out_of_range":
        feedback = (
            "Invalid response: frame indices out of range. "
            f"Valid range is [0, {max(0, int(frame_count or 0) - 1)}]. "
            "Choose NEW frame indices within range and not already seen."
        )
    elif reason == "frames_not_in_candidates":
        suffix = f" Candidate ranges: {candidate_text}." if candidate_text else ""
        feedback = (
            "Invalid response: requested frames must be chosen ONLY as individual indices from the candidate unseen "
            "frame intervals."
            + suffix
        )
    elif reason == "empty_frames":
        feedback = (
            f"Invalid response: <select> is empty. Provide 1-{k} NEW frame indices to view NEXT."
        )
    elif reason == "frames_all_seen":
        feedback = (
            "Invalid response: all requested frames are already seen. "
            f"In <select>, choose individual indices: output 1-{k} NEW indices NOT in the Seen frames list. "
            f"Allowed unseen intervals: {unseen_text}. Do NOT output ranges like 1-4."
        )
    elif reason == "frames_already_seen":
        feedback = (
            "Invalid response: requested frames include frames that were already seen. "
            f"In <select>, choose individual indices: output 1-{k} NEW indices NOT in the Seen frames list. "
            f"Allowed unseen intervals: {unseen_text}. Do NOT output ranges like 1-4."
        )
    elif reason == "too_many_frames":
        feedback = (
            f"Invalid response: requested too many NEW frames. Choose at most {k} "
            "NEW frame indices not already in Seen frames."
        )
    elif reason == "invalid_frames":
        feedback = (
            "Invalid response: requested frames must be NEW and within range. "
            f"In <select>, output 1-{k} comma-separated individual integers NOT in Seen frames. "
            f"Allowed unseen intervals: {unseen_text}. Do NOT output ranges like 1-4."
        )
    else:
        feedback = str(reason)

    return retry_feedback_text(feedback, force_answer=force_answer, force_instructions=force_instructions)
