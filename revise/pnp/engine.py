"""Shared plug-and-play multi-round video-QA loop."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from revise.pnp.policy import build_retry_feedback, decide_revise_action, resolve_invalid_revise_action
from revise.pnp.protocols import Backend, Dataset, LoopConfig, RunStats
from revise.pnp.utils import (
    maybe_log_jsonl,
    truncate_text,
)


@dataclass
class SampleOutcome:
    answer_letter: Optional[str]
    seen_frames: list[int]
    round_idx: int
    effective_rounds: int
    terminated_reason: Optional[str]
    terminated_invalid_action: bool
    answer_frame_count: int = 0


@dataclass
class OneshotOutcome:
    """Result of a single-round (one-shot) baseline sample.

    ``failed_reason`` is ``None`` on a clean single chat; otherwise the model was
    never queried (e.g. an empty timeline or no extracted frames) and the harness
    counts the sample as failed without logging a prediction.
    """

    answer_letter: Optional[str]
    raw_output: Optional[str]
    frame_indices: list[int]
    failed_reason: Optional[str] = None


def run_sample_oneshot(
    sample: Any,
    *,
    dataset: Dataset,
    backend: Backend,
    cfg: LoopConfig,
    stats: RunStats,
    rng: random.Random,
    base_url: str,
    model_id: str,
    run: Any = None,
) -> OneshotOutcome:
    """Single-round baseline: sample frames once, one chat, parse the answer.

    Deliberately separate from the multi-round :func:`run_sample`. It reuses the
    same Dataset/Backend adapter methods (frame_count / initial_frame_indices /
    extract_frames / format_question / normalize_answer) plus the one-shot-only
    ``oneshot_user_text``. No select/summarize/retry loop.
    """
    _ = run, rng

    try:
        frame_count = dataset.frame_count(sample)
    except Exception:
        return OneshotOutcome(
            answer_letter=None, raw_output=None, frame_indices=[], failed_reason="invalid_video_timeline"
        )
    if frame_count <= 0:
        return OneshotOutcome(
            answer_letter=None, raw_output=None, frame_indices=[], failed_reason="invalid_video_timeline"
        )

    question_block = dataset.format_question(sample)
    frame_indices = [int(i) for i in dataset.initial_frame_indices(sample, frame_count, cfg)]
    images = dataset.extract_frames(sample, frame_indices)
    if not images:
        return OneshotOutcome(
            answer_letter=None, raw_output=None, frame_indices=frame_indices, failed_reason="no_frames"
        )

    user_text = dataset.oneshot_user_text(question_block, len(images), frame_indices=frame_indices)
    raw = backend.chat(
        base_url=base_url,
        model_id=model_id,
        system_prompt="",
        user_text=user_text,
        images=images,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        timeout_s=cfg.request_timeout_s,
    )
    stats.total_model_calls += 1
    answer_letter = dataset.normalize_answer(sample, raw)
    return OneshotOutcome(
        answer_letter=answer_letter,
        raw_output=raw,
        frame_indices=frame_indices,
        failed_reason=None,
    )


_LOG_PREFIX_FIELDS = ("sample_id", "qid", "video_id", "video_path")
_LOG_QUESTION_FIELDS = ("question", "choices", "ground_truth_idx")


def _select_log_fields(fields: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: fields[name] for name in names if name in fields}


def _extra_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    known = set(_LOG_PREFIX_FIELDS) | set(_LOG_QUESTION_FIELDS)
    return {name: value for name, value in fields.items() if name not in known}


def run_sample(
    sample: Any,
    *,
    dataset: Dataset,
    backend: Backend,
    cfg: LoopConfig,
    stats: RunStats,
    rng: random.Random,
    base_url: str,
    model_id: str,
    run: Any = None,
) -> SampleOutcome:
    _ = run
    seen_frames: list[int] = []
    video_path = dataset.video_path(sample)
    video_id = dataset.video_id(sample)
    sample_frame_count = dataset.frame_count(sample)
    frame_count = sample_frame_count
    sample_log_fields = dataset.log_fields(sample)
    if frame_count <= 0 and getattr(cfg, "observation_mode", "image") != "caption":
        try:
            import decord

            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            frame_count = int(len(vr))
        except Exception:
            frame_count = 0

    question_block = dataset.format_question(sample)
    system_prompt = dataset.system_prompt(cfg)

    video_captions: dict[int, str] = {}
    if getattr(cfg, "captions_dir", None) and getattr(cfg, "caption_include", "none") != "none":
        video_captions = dataset.load_video_captions(str(cfg.captions_dir), video_id)

    summary_state = dataset.initial_summary(cfg)
    effective_rounds = 0
    terminated_reason: Optional[str] = None
    terminated_invalid_action = False

    # Caption-only mode uses caption indices (1fps) as the action space length L.
    observation_mode = getattr(cfg, "observation_mode", "image")
    fps = 0.0
    if observation_mode == "caption":
        if video_captions:
            frame_count = max(video_captions.keys(), default=-1) + 1
        if frame_count <= 0:
            # Fall back to a rough seconds estimate from video length, if available.
            try:
                import decord

                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                video_len = int(len(vr))
                fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
                if fps and fps > 0 and video_len > 0:
                    frame_count = max(1, int(video_len / fps))
            except Exception:
                frame_count = max(1, int(sample_frame_count) if int(sample_frame_count) > 0 else 1)
    elif video_captions:
        fps = dataset.get_video_fps(video_path)

    def _caption_for_index(idx: int) -> str:
        if not video_captions:
            return "[no caption]"
        key = int(idx)
        if observation_mode != "caption":
            key = dataset.caption_key_for_frame_index(int(idx), fps)
        return video_captions.get(int(key)) or "[no caption]"

    init_frames = dataset.initial_frame_indices(sample, frame_count, cfg)
    next_frames = [int(i) for i in init_frames if i >= 0]
    answer_letter: Optional[str] = None
    answer_frame_count = 0

    def _count_exhausted_invalid_as_retry(reason: str) -> None:
        if dataset.should_count_exhausted_invalid_as_retry(reason):
            stats.total_retries += 1

    def _should_clear_frame_plan(reason: str) -> bool:
        return dataset.should_clear_frame_plan_on_exhausted_invalid(reason)

    for round_idx in range(1, cfg.max_rounds + 1):
        # Frames shown in this round.
        frames_this_round = [i for i in next_frames if i not in seen_frames]
        if not frames_this_round:
            frames_this_round = dataset.fallback_frame_indices(sample, frame_count, 1, cfg)
        frames_this_round = frames_this_round[: cfg.max_frames_per_round]
        for i in frames_this_round:
            if i not in seen_frames:
                seen_frames.append(i)

        candidate_next_frames: list[int] = []
        if getattr(cfg, "use_candidate_frames", False):
            k = cfg.candidate_k if cfg.candidate_k is not None else max(12, cfg.max_frames_per_round * 4)
            candidate_next_frames = dataset.candidate_frame_indices(
                sample,
                frame_count=frame_count,
                seen_frames=seen_frames,
                k=k,
                rng=rng,
            )
        shown_captions: Optional[list[str]] = None
        candidate_captions: Optional[list[str]] = None
        shown_ts: Optional[list[int]] = None
        candidate_ts: Optional[list[int]] = None
        if video_captions:
            include = getattr(cfg, "caption_include", "none")
            max_chars = int(getattr(cfg, "caption_max_chars", 0))
            if include in ("shown", "both"):
                shown_captions = [
                    truncate_text(_caption_for_index(int(i)), max_chars)
                    for i in frames_this_round
                ]
                if observation_mode != "caption":
                    shown_ts = [dataset.caption_key_for_frame_index(int(i), fps) for i in frames_this_round]
            if include in ("candidate", "both") and candidate_next_frames:
                candidate_captions = [
                    truncate_text(_caption_for_index(int(i)), max_chars)
                    for i in candidate_next_frames
                ]
                if observation_mode != "caption":
                    candidate_ts = [dataset.caption_key_for_frame_index(int(i), fps) for i in candidate_next_frames]
        images: list[Any] = []
        if observation_mode != "caption":
            images = dataset.extract_frames(sample, frames_this_round)
            if dataset.should_fail_on_empty_images(cfg) and not images:
                raise RuntimeError("No frames extracted for image-mode sample.")
        user_text = dataset.build_user_text(
            question_block=question_block,
            summary=summary_state,
            frame_count=frame_count,
            round_idx=round_idx,
            frame_indices=frames_this_round,
            seen_frames=seen_frames,
            render_images=(observation_mode != "caption"),
            hide_seen_frames=bool(getattr(cfg, "hide_seen_frames_in_prompt", False)),
            candidate_unseen_frames=candidate_next_frames if getattr(cfg, "use_candidate_frames", False) else None,
            use_candidate_frame_ids=bool(cfg.use_candidate_frame_ids),
            require_candidate_frames=bool(getattr(cfg, "require_candidate_frames", False)),
            shown_frame_captions=shown_captions,
            candidate_id_captions=candidate_captions,
            shown_frame_ts=shown_ts,
            candidate_id_ts=candidate_ts,
        )
        if cfg.force_final_answer and round_idx >= cfg.max_rounds:
            final_round_instruction = dataset.final_round_instruction(cfg)
            if final_round_instruction:
                user_text = f"{user_text}\n\n{final_round_instruction}"
        raw = ""
        retry_feedback: Optional[str] = None
        attempt_user_text = user_text
        min_select_rounds = min(max(0, int(cfg.min_select_rounds or 0)), max(0, int(cfg.max_rounds) - 1))
        should_answer_this_round = bool(cfg.force_final_answer and round_idx >= cfg.max_rounds)
        for retry_idx in range(max(0, int(cfg.max_retries_per_round)) + 1):
            raw = backend.chat(
                base_url=base_url,
                model_id=model_id,
                system_prompt=system_prompt,
                user_text=attempt_user_text,
                images=images,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                timeout_s=cfg.request_timeout_s,
            )
            stats.total_model_calls += 1

            frames_tag = dataset.parse_select(raw)
            requested_raw_frames: Optional[list[int]] = None
            requested_mapped_frames: Optional[list[int]] = None
            if frames_tag is not None:
                requested_raw_frames = dataset.parse_select_frames(frames_tag)
                if bool(cfg.use_candidate_frame_ids) and candidate_next_frames:
                    requested_mapped_frames = dataset.map_candidate_frame_ids(
                        requested_raw_frames,
                        candidate_next_frames,
                    )
                else:
                    requested_mapped_frames = requested_raw_frames
            decision = decide_revise_action(
                raw,
                num_choices=dataset.num_choices(sample),
                seen_frames=seen_frames,
                frame_count=frame_count,
                max_frames_per_round=cfg.max_frames_per_round,
                round_idx=round_idx,
                max_rounds=cfg.max_rounds,
                answer_only_final_round=bool(cfg.answer_only_final_round),
                require_answer=should_answer_this_round,
                min_select_rounds=min_select_rounds,
                select_rounds_so_far=effective_rounds,
                candidate_frames=candidate_next_frames,
                use_candidate_frame_ids=bool(cfg.use_candidate_frame_ids),
                require_candidate_frames=bool(getattr(cfg, "require_candidate_frames", False)),
                parse_select_frames=dataset.parse_select_frames,
                normalize_answer=lambda answer: dataset.normalize_answer(sample, answer),
                is_select_summary_valid=lambda summary, seen_count: dataset.is_select_summary_valid(
                    summary, seen_count=seen_count
                ),
                is_summary_stale=lambda summary, seen_count: dataset.is_summary_stale(
                    summary, seen_count=seen_count
                ),
                select_has_range_syntax=dataset.select_has_range_syntax,
            )
            raw_answer = dataset.parse_answer(raw)
            raw_answer_letter = dataset.normalize_answer(sample, raw_answer) if raw_answer else None
            logged_answer_letter = (
                decision.answer_letter
                if decision.kind == "answer"
                else (raw_answer_letter if not cfg.strict_actions else None)
            )
            accepted_answer_letter = decision.answer_letter if decision.kind == "answer" else None

            maybe_log_jsonl(
                cfg.log_jsonl,
                {
                    "ts": time.time(),
                    **_select_log_fields(sample_log_fields, _LOG_PREFIX_FIELDS),
                    "round_idx": round_idx,
                    "retry_idx": retry_idx,
                    "retry_feedback": retry_feedback,
                    **_select_log_fields(sample_log_fields, _LOG_QUESTION_FIELDS),
                    **_extra_log_fields(sample_log_fields),
                    "observation_mode": observation_mode,
                    "use_candidate_frames": bool(getattr(cfg, "use_candidate_frames", False)),
                    "use_candidate_frame_ids": bool(cfg.use_candidate_frame_ids),
                    "candidate_unseen_frames": (
                        candidate_next_frames if getattr(cfg, "use_candidate_frames", False) else None
                    ),
                    "captions_dir": getattr(cfg, "captions_dir", None),
                    "caption_include": getattr(cfg, "caption_include", "none"),
                    "caption_max_chars": int(getattr(cfg, "caption_max_chars", 0)),
                    "shown_frame_captions": shown_captions,
                    "candidate_id_captions": candidate_captions,
                    "seen_frames": seen_frames,
                    "current_frames": frames_this_round,
                    "requested_raw_frames": requested_raw_frames,
                    "requested_mapped_frames": requested_mapped_frames,
                    "summary_in": summary_state,
                    "system_prompt": system_prompt,
                    "user_text": attempt_user_text,
                    "raw_output": raw,
                    "action_kind": decision.kind,
                    "invalid_reason": decision.reason if decision.kind == "invalid" else None,
                    "raw_answer_letter": raw_answer_letter,
                    "answer_letter": logged_answer_letter,
                    "accepted_answer_letter": accepted_answer_letter,
                },
            )
            if decision.think is not None and hasattr(dataset, "think_present"):
                dataset.think_present += 1

            if decision.kind == "invalid":
                reason = decision.reason or "invalid_paper_protocol"
                stats.invalid_outputs += 1
                terminated_reason = reason
                invalid_resolution = resolve_invalid_revise_action(
                    reason,
                    retry_idx=retry_idx,
                    max_retries_per_round=cfg.max_retries_per_round,
                    retryable=dataset.should_retry_invalid_output(reason),
                    strict_actions=cfg.strict_actions,
                    clear_frame_plan=_should_clear_frame_plan(reason),
                )
                if invalid_resolution.kind == "retry":
                    stats.total_retries += 1
                    retry_feedback = build_retry_feedback(
                        reason,
                        force_answer=should_answer_this_round,
                        max_frames_per_round=cfg.max_frames_per_round,
                        frame_count=frame_count,
                        seen_frames=seen_frames,
                        num_choices=dataset.num_choices(sample),
                        candidate_count=len(candidate_next_frames),
                        candidate_frames=candidate_next_frames,
                    )
                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                    continue

                _count_exhausted_invalid_as_retry(reason)
                if invalid_resolution.kind == "terminate":
                    stats.invalid_action_terminated += 1
                    terminated_invalid_action = True
                    answer_letter = None
                    break
                if invalid_resolution.kind == "clear_frame_plan":
                    next_frames = []
                    break

                stats.fallback_frames_used += 1
                requested = (
                    candidate_next_frames[: cfg.max_frames_per_round]
                    if candidate_next_frames
                    else dataset.sample_unseen_frames(frame_count, set(seen_frames), cfg.max_frames_per_round, rng=rng)
                )
                next_frames = (
                    requested[: cfg.max_frames_per_round]
                    if requested
                    else dataset.fallback_frame_indices(sample, frame_count, 1, cfg)
                )
                break

            if decision.kind == "answer":
                answer_letter = decision.answer_letter
                answer_frame_count = len(frames_this_round)
                break

            summary = decision.summary
            if dataset.should_commit_summary(summary, seen_count=len(seen_frames)):
                summary_state = str(summary)
            next_frames = list(decision.requested_frames)[: cfg.max_frames_per_round]
            effective_rounds += 1
            stats.effective_rounds_total += 1
            break

        if answer_letter is not None:
            break
        if cfg.strict_actions and terminated_invalid_action:
            break

    stats.total_rounds += round_idx

    return SampleOutcome(
        answer_letter=answer_letter,
        seen_frames=seen_frames,
        round_idx=round_idx,
        effective_rounds=effective_rounds,
        terminated_reason=terminated_reason,
        terminated_invalid_action=terminated_invalid_action,
        answer_frame_count=answer_frame_count,
    )
