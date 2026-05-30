"""Shared plug-and-play multi-round video-QA loop."""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from revise.pnp_protocols import Backend, Dataset, LoopConfig, RunStats
from revise.pnp_utils import (
    ANSWER_RE,
    SELECT_RE,
    SUMMARIZE_RE,
    THINK_RE,
    contains_banned_example,
    dedupe_preserve_order,
    extract_tag,
    format_intervals,
    in_intervals,
    is_placeholder,
    maybe_log_jsonl,
    normalize_answer_letter,
    parse_int_list,
    propose_candidate_frames,
    sample_uniform_indices,
    summary_has_ohrpu,
    truncate_text,
    unseen_intervals,
)


@dataclass
class SampleOutcome:
    answer_letter: Optional[str]
    seen_frames: list[int]
    round_idx: int
    effective_rounds: int
    terminated_reason: Optional[str]
    terminated_invalid_action: bool


def _frames_has_range_syntax(frames_text: str) -> bool:
    if not frames_text:
        return False
    # Common failure mode: model copies allowed ranges like "4-182" into <select>.
    return bool(re.search(r"\d+\s*[-–—]\s*\d+", frames_text))


def _summary_has_stale_boilerplate(summary_text: str, *, seen_count: int) -> bool:
    if not summary_text or seen_count <= 0:
        return False
    s = re.sub(r"\s+", " ", str(summary_text)).strip().lower()
    if "has not seen any frames yet" in s:
        return True
    if re.search(r"\bhas not seen any (frame|frames|caption|captions) yet\b", s):
        return True
    if "no frames yet" in s:
        return True
    if "no captions yet" in s:
        return True
    return False


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
    frame_count = sample.frame_count
    if frame_count <= 0 and getattr(cfg, "observation_mode", "image") != "caption":
        try:
            import decord

            vr = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))
            frame_count = int(len(vr))
        except Exception:
            frame_count = 0

    question_block = dataset.format_question(sample)
    system_prompt = dataset.system_prompt(cfg)

    video_captions: dict[int, str] = {}
    if getattr(cfg, "captions_dir", None) and getattr(cfg, "caption_include", "none") != "none":
        video_captions = dataset.load_video_captions(str(cfg.captions_dir), sample.video_id)

    summary_state = (
        "P: I will summarize what has been shown so far; "
        "O: I will record the key observations from the current evidence; "
        "H: I will update my belief as new evidence arrives; "
        "U: some key detail may still be unclear; "
        "R: request more evidence if needed"
    )
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

                vr = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))
                video_len = int(len(vr))
                fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
                if fps and fps > 0 and video_len > 0:
                    frame_count = max(1, int(video_len / fps))
            except Exception:
                frame_count = max(1, int(sample.frame_count) if int(sample.frame_count) > 0 else 1)
    elif video_captions:
        fps = dataset.get_video_fps(sample.video_path)

    def _caption_for_index(idx: int) -> str:
        if not video_captions:
            return "[no caption]"
        key = int(idx)
        if observation_mode != "caption":
            key = dataset.caption_key_for_frame_index(int(idx), fps)
        return video_captions.get(int(key)) or "[no caption]"

    init_frames = sample_uniform_indices(frame_count, cfg.max_frames_per_round)
    next_frames = [int(i) for i in init_frames if i >= 0]
    answer_letter: Optional[str] = None
    last_user_text: Optional[str] = None
    last_images: list[Any] = []
    last_frames: list[int] = []

    for round_idx in range(1, cfg.max_rounds + 1):
        # Frames shown in this round.
        frames_this_round = [i for i in next_frames if i not in seen_frames]
        if not frames_this_round:
            frames_this_round = sample_uniform_indices(frame_count, 1)
        frames_this_round = frames_this_round[: cfg.max_frames_per_round]
        for i in frames_this_round:
            if i not in seen_frames:
                seen_frames.append(i)

        candidate_next_frames: list[int] = []
        if getattr(cfg, "use_candidate_frames", False):
            k = cfg.candidate_k if cfg.candidate_k is not None else max(12, cfg.max_frames_per_round * 4)
            candidate_next_frames = propose_candidate_frames(
                frame_count=frame_count,
                seen=set(seen_frames),
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
            user_text = (
                f"{user_text}\n\n"
                "This is the final round. You MUST answer now using <think>...</think> then <answer>LETTER</answer>."
            )
        last_user_text = user_text
        last_images = images
        last_frames = frames_this_round

        raw = ""
        retry_feedback: Optional[str] = None
        attempt_user_text = user_text
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

            frames_tag = extract_tag(raw, SELECT_RE)
            requested_raw_frames: Optional[list[int]] = None
            requested_mapped_frames: Optional[list[int]] = None
            if frames_tag is not None:
                requested_raw_frames = dedupe_preserve_order(parse_int_list(frames_tag))
                if bool(cfg.use_candidate_frame_ids) and candidate_next_frames:
                    mapped: list[int] = []
                    invalid_id = False
                    for cid in requested_raw_frames:
                        if 1 <= cid <= len(candidate_next_frames):
                            mapped.append(int(candidate_next_frames[cid - 1]))
                        else:
                            invalid_id = True
                    requested_mapped_frames = None if invalid_id else dedupe_preserve_order(mapped)
                else:
                    requested_mapped_frames = requested_raw_frames

            maybe_log_jsonl(
                cfg.log_jsonl,
                {
                    "ts": time.time(),
                    "sample_id": sample.sample_id,
                    "qid": sample.qid,
                    "video_id": sample.video_id,
                    "video_path": sample.video_path,
                    "round_idx": round_idx,
                    "retry_idx": retry_idx,
                    "retry_feedback": retry_feedback,
                    "question": sample.question,
                    "choices": sample.choices,
                    "ground_truth_idx": sample.answer_idx,
                    "observation_mode": observation_mode,
                    "use_candidate_frames": bool(getattr(cfg, "use_candidate_frames", False)),
                    "use_candidate_frame_ids": bool(cfg.use_candidate_frame_ids),
                    "candidate_unseen_frames": candidate_next_frames if getattr(cfg, "use_candidate_frames", False) else None,
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
                },
            )

            summary = extract_tag(raw, SUMMARIZE_RE)
            if (
                summary
                and (not is_placeholder(summary))
                and (not contains_banned_example(summary))
                and summary_has_ohrpu(summary)
                and (not _summary_has_stale_boilerplate(summary, seen_count=len(seen_frames)))
            ):
                summary_state = summary

            think = extract_tag(raw, THINK_RE)
            if think is None:
                stats.invalid_outputs += 1
                terminated_reason = "missing_think"
                if retry_idx < int(cfg.max_retries_per_round):
                    stats.total_retries += 1
                    retry_feedback = dataset.retry_feedback_text(
                        "Invalid response: every response MUST begin with a <think>...</think> reasoning trace, "
                        "then either <summarize> + <select> (request) or <answer> (final).",
                        force_answer=bool(cfg.force_final_answer and round_idx >= cfg.max_rounds),
                    )
                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                    continue
                if cfg.strict_actions:
                    stats.invalid_action_terminated += 1
                    terminated_invalid_action = True
                    answer_letter = None
                    break
                # Fall back to the usual next_frames heuristic.
                stats.fallback_frames_used += 1
                requested = dataset.sample_unseen_frames(
                    frame_count, set(seen_frames), cfg.max_frames_per_round, rng=rng
                )
                next_frames = (
                    requested[: cfg.max_frames_per_round]
                    if requested
                    else sample_uniform_indices(frame_count, 1)
                )
                break

            answer = extract_tag(raw, ANSWER_RE)
            if answer:
                answer_letter = normalize_answer_letter(answer, len(sample.choices))
                if answer_letter is None:
                    stats.invalid_outputs += 1
                    terminated_reason = "invalid_answer_letter"
                    if retry_idx < int(cfg.max_retries_per_round):
                        stats.total_retries += 1
                        retry_feedback = dataset.retry_feedback_text(
                            "Invalid response: <answer> must be exactly ONE option letter (A/B/C/D/E). "
                            "Do not output words or a sentence.",
                            force_answer=True,
                        )
                        attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                        continue
                    if cfg.strict_actions:
                        stats.invalid_action_terminated += 1
                        terminated_invalid_action = True
                        answer_letter = None
                        break
                    # Non-strict: ignore the invalid answer and continue with a fallback frame.
                    stats.fallback_frames_used += 1
                    next_frames = dataset.sample_unseen_frames(
                        frame_count, set(seen_frames), cfg.max_frames_per_round, rng=rng
                    )
                    if not next_frames:
                        next_frames = sample_uniform_indices(frame_count, 1)
                    answer_letter = None
                    break

                if bool(cfg.answer_only_final_round) and round_idx < cfg.max_rounds:
                    stats.invalid_outputs += 1
                    terminated_reason = "early_answer_disallowed"
                    if retry_idx < int(cfg.max_retries_per_round):
                        stats.total_retries += 1
                        retry_feedback = dataset.retry_feedback_text(
                            "Invalid response: do NOT answer yet. Request more frames using <summarize>...</summarize> and <select>...</select>.",
                            force_answer=False,
                        )
                        attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                        answer_letter = None
                        continue
                    if cfg.strict_actions:
                        stats.invalid_action_terminated += 1
                        terminated_invalid_action = True
                        answer_letter = None
                        break
                    # Non-strict: ignore the early answer and continue with a fallback frame request.
                    stats.fallback_frames_used += 1
                    next_frames = dataset.sample_unseen_frames(
                        frame_count, set(seen_frames), cfg.max_frames_per_round, rng=rng
                    )
                    if not next_frames:
                        next_frames = sample_uniform_indices(frame_count, 1)
                    answer_letter = None
                    break

                # Strict-paper Answer round: <think> + <answer> only. No <summarize>
                # is required here; the last committed summary is reused as the state
                # (captured above when a valid <summarize> was present on a Select round).
                break

            frames_text = extract_tag(raw, SELECT_RE)

            # If we didn't answer, we must request frames, with a valid summary.
            if frames_text is None:
                stats.invalid_outputs += 1
                terminated_reason = "missing_frames_tag"
                if retry_idx < int(cfg.max_retries_per_round):
                    stats.total_retries += 1
                    retry_feedback = dataset.retry_feedback_text(
                        "Invalid response: missing <select> tag for requesting more frames. "
                        "Remember: <select> must list NEW frame indices to view NEXT (not already seen).",
                        force_answer=bool(cfg.force_final_answer and round_idx >= cfg.max_rounds),
                    )
                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                    continue
                if cfg.strict_actions:
                    stats.invalid_action_terminated += 1
                    terminated_invalid_action = True
                    answer_letter = None
                    break
                next_frames = sample_uniform_indices(frame_count, 1)
                break

            if summary is None or is_placeholder(summary) or contains_banned_example(summary) or (not summary_has_ohrpu(summary)):
                stats.invalid_outputs += 1
                terminated_reason = "invalid_select_summary"
                if retry_idx < int(cfg.max_retries_per_round):
                    stats.total_retries += 1
                    retry_feedback = dataset.retry_feedback_text(
                        "Invalid response: include a meaningful <summarize> with P/O/H/U/R in that exact order "
                        "(no placeholders like '.../none/unknown').",
                        force_answer=bool(cfg.force_final_answer and round_idx >= cfg.max_rounds),
                    )
                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                    continue
            if summary is not None and _summary_has_stale_boilerplate(summary, seen_count=len(seen_frames)):
                stats.invalid_outputs += 1
                terminated_reason = "stale_select_summary"
                if retry_idx < int(cfg.max_retries_per_round):
                    stats.total_retries += 1
                    retry_feedback = dataset.retry_feedback_text(
                        "Invalid response: the <summarize> claims no frames/captions were seen, but evidence was shown. "
                        "Rewrite <summarize> to reflect what was observed so far (P/O/H/U/R), then request frames.",
                        force_answer=bool(cfg.force_final_answer and round_idx >= cfg.max_rounds),
                    )
                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                    continue

            if (not bool(cfg.use_candidate_frame_ids)) and _frames_has_range_syntax(frames_text):
                stats.invalid_outputs += 1
                terminated_reason = "frames_range_syntax"
                if retry_idx < int(cfg.max_retries_per_round):
                    stats.total_retries += 1
                    retry_feedback = dataset.retry_feedback_text(
                        "Invalid response: <select> must be a comma-separated list of integers only "
                        "(NO ranges like '4-182', no hyphens). Choose up to {k} NEW frames.".format(
                            k=cfg.max_frames_per_round
                        ),
                        force_answer=bool(cfg.force_final_answer and round_idx >= cfg.max_rounds),
                    )
                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                    continue

            requested = dedupe_preserve_order(parse_int_list(frames_text))
            if bool(cfg.use_candidate_frame_ids) and candidate_next_frames:
                mapped: list[int] = []
                invalid_id = False
                for cid in requested:
                    if 1 <= cid <= len(candidate_next_frames):
                        mapped.append(int(candidate_next_frames[cid - 1]))
                    else:
                        invalid_id = True
                if invalid_id:
                    stats.invalid_outputs += 1
                    terminated_reason = "frames_out_of_range"
                    if retry_idx < int(cfg.max_retries_per_round):
                        stats.total_retries += 1
                        retry_feedback = dataset.retry_feedback_text(
                            "Invalid response: when Candidate Frame IDs are provided, <select> must contain only "
                            "IDs in the allowed range [1..K] (comma-separated integers).",
                            force_answer=bool(cfg.force_final_answer and round_idx >= cfg.max_rounds),
                        )
                        attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                        continue
                    if cfg.strict_actions and not bool(getattr(cfg, "fallback_on_invalid_candidate_ids", True)):
                        stats.invalid_action_terminated += 1
                        terminated_invalid_action = True
                        answer_letter = None
                        break
                    # Be forgiving: fall back to heuristic sampling instead of hard-terminating.
                    stats.fallback_frames_used += 1
                    requested = candidate_next_frames[: cfg.max_frames_per_round]
                    if not requested:
                        requested = dataset.sample_unseen_frames(
                            frame_count, set(seen_frames), cfg.max_frames_per_round, rng=rng
                        )
                    next_frames = (
                        requested[: cfg.max_frames_per_round]
                        if requested
                        else sample_uniform_indices(frame_count, 1)
                    )
                    break
                else:
                    requested = dedupe_preserve_order(mapped)
                    requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames]
            else:
                if bool(getattr(cfg, "require_candidate_frames", False)) and candidate_next_frames:
                    allowed = {int(i) for i in candidate_next_frames}
                    disallowed = [i for i in requested if int(i) not in allowed]
                    if disallowed:
                        stats.invalid_outputs += 1
                        terminated_reason = "frames_not_in_candidates"
                        if retry_idx < int(cfg.max_retries_per_round):
                            stats.total_retries += 1
                            retry_feedback = dataset.retry_feedback_text(
                                "Invalid response: requested frames must be chosen ONLY from the candidate unseen frame list/ranges provided.",
                                force_answer=bool(cfg.force_final_answer and round_idx >= cfg.max_rounds),
                            )
                            attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                            continue
                        if cfg.strict_actions:
                            stats.invalid_action_terminated += 1
                            terminated_invalid_action = True
                            answer_letter = None
                            break
                        requested = []
                    else:
                        requested = [i for i in requested if 0 <= i < frame_count and i not in seen_frames and int(i) in allowed]
                else:
                    allowed_ranges = unseen_intervals(frame_count, seen_frames)
                    requested = [
                        i
                        for i in requested
                        if 0 <= i < frame_count and i not in seen_frames and in_intervals(i, allowed_ranges)
                    ]

            if requested and len(requested) > int(cfg.max_frames_per_round):
                stats.invalid_outputs += 1
                terminated_reason = "too_many_frames"
                requested = requested[: int(cfg.max_frames_per_round)]
            if not requested:
                stats.invalid_outputs += 1
                terminated_reason = "invalid_frames"
                if retry_idx < int(cfg.max_retries_per_round):
                    stats.total_retries += 1
                    candidate_text = (
                        " Allowed unseen ranges: "
                        f"{format_intervals(unseen_intervals(frame_count, seen_frames))}."
                    )
                    retry_feedback = dataset.retry_feedback_text(
                        "Invalid response: requested frames must be NEW and within range. "
                        "In <select>, output 1–{k} comma-separated integers NOT in Seen frames.".format(
                            k=cfg.max_frames_per_round
                        )
                        + candidate_text,
                        force_answer=bool(cfg.force_final_answer and round_idx >= cfg.max_rounds),
                    )
                    attempt_user_text = f"{user_text}\n\n{retry_feedback}"
                    continue
                if cfg.strict_actions:
                    stats.invalid_action_terminated += 1
                    terminated_invalid_action = True
                    answer_letter = None
                    break

                # Fall back to heuristic sampling.
                stats.fallback_frames_used += 1
                requested = candidate_next_frames[: cfg.max_frames_per_round]
                if not requested:
                    requested = dataset.sample_unseen_frames(
                        frame_count, set(seen_frames), cfg.max_frames_per_round, rng=rng
                    )
                next_frames = (
                    requested[: cfg.max_frames_per_round] if requested else sample_uniform_indices(frame_count, 1)
                )
                break

            next_frames = requested[: cfg.max_frames_per_round]
            effective_rounds += 1
            stats.effective_rounds_total += 1
            break

        if answer_letter is not None:
            break
        if cfg.strict_actions and terminated_invalid_action:
            break

    stats.total_rounds += round_idx
    if (
        cfg.force_final_answer
        and answer_letter is None
        and last_user_text is not None
        and not (cfg.strict_actions and terminated_invalid_action)
    ):
        forced_user_text = (
            f"{last_user_text}\n\n"
            "Max rounds reached. Provide the final answer now using <think>...</think> then <answer>LETTER</answer>."
        )
        raw = backend.chat(
            base_url=base_url,
            model_id=model_id,
            system_prompt=system_prompt,
            user_text=forced_user_text,
            images=last_images,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.request_timeout_s,
        )
        stats.total_model_calls += 1
        maybe_log_jsonl(
            cfg.log_jsonl,
            {
                "ts": time.time(),
                "sample_id": sample.sample_id,
                "qid": sample.qid,
                "video_id": sample.video_id,
                "video_path": sample.video_path,
                "round_idx": cfg.max_rounds + 1,
                "forced_answer": True,
                "question": sample.question,
                "choices": sample.choices,
                "ground_truth_idx": sample.answer_idx,
                "seen_frames": seen_frames,
                "current_frames": last_frames,
                "summary_in": summary_state,
                "system_prompt": system_prompt,
                "user_text": forced_user_text,
                "raw_output": raw,
            },
        )
        answer = extract_tag(raw, ANSWER_RE)
        if answer:
            answer_letter = normalize_answer_letter(answer, len(sample.choices))

    return SampleOutcome(
        answer_letter=answer_letter,
        seen_frames=seen_frames,
        round_idx=round_idx,
        effective_rounds=effective_rounds,
        terminated_reason=terminated_reason,
        terminated_invalid_action=terminated_invalid_action,
    )
