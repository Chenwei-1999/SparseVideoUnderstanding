# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""REVISE-style multi-round agent loop for sparse video QA."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from PIL import Image

from examples.revise.pnp_prompts import SYSTEM_PROMPT
from examples.revise.pnp_utils import (
    ANSWER_RE,
    SELECT_RE,
    SUMMARIZE_RE,
    THINK_RE,
    contains_banned_example,
    dedupe_preserve_order,
    extract_tag,
    format_frame_list,
    format_intervals,
    in_intervals,
    indices_to_intervals,
    is_placeholder,
    normalize_answer_letter,
    parse_int_list,
    propose_candidate_frames,
    sample_uniform_indices,
    summary_has_ohrpu,
    unseen_intervals,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Re-export under old private names for backward compatibility within this module.
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
_SUMMARIZE_RE = SUMMARIZE_RE
_SELECT_RE = SELECT_RE
_ANSWER_RE = ANSWER_RE
_THINK_RE = THINK_RE
_contains_banned_example = contains_banned_example
_dedupe_preserve_order = dedupe_preserve_order
_extract_tag = extract_tag
_format_frame_list = format_frame_list
_format_intervals = format_intervals
_in_intervals = in_intervals
_indices_to_intervals = indices_to_intervals
_is_placeholder = is_placeholder
_normalize_answer_letter = normalize_answer_letter
_parse_frame_indices = parse_int_list
_summary_has_ohrpu = summary_has_ohrpu
_unseen_intervals = unseen_intervals


def _sample_uniform_indices(frame_count: int, n: int) -> list[int]:
    """Agent-loop variant using numpy (matches training-time behavior)."""
    if frame_count <= 0:
        return list(range(n))
    if n <= 1:
        return [frame_count // 2]
    return [int(x) for x in np.linspace(0, frame_count - 1, n)]


def _propose_candidate_unseen_frames(
    frame_count: int,
    seen: set[int],
    k: int,
    rng: random.Random,
) -> list[int]:
    """Alias for the shared propose_candidate_frames."""
    return propose_candidate_frames(frame_count, seen, k, rng)


def _load_video_meta(video_path: str) -> tuple[Optional[int], Optional[float]]:
    # Returns (frame_count, fps)
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frame_count = _safe_nonnegative_int(len(vr))
        fps = _safe_positive_float(vr.get_avg_fps())
        if frame_count is not None or fps is not None:
            return frame_count, fps
    except Exception:
        pass

    try:
        import imageio

        reader = imageio.get_reader(video_path, "ffmpeg")
        meta = reader.get_meta_data()
        fps = _safe_positive_float(meta.get("fps"))
        nframes = _safe_nonnegative_int(meta.get("nframes"))
        if nframes is None:
            duration = _safe_positive_float(meta.get("duration"))
            if duration is not None and fps is not None:
                nframes = max(1, int(math.ceil(duration * fps)))
        reader.close()
        return nframes, fps
    except Exception:
        return None, None


def _safe_nonnegative_int(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return max(0, int(number))


def _safe_positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _timeline_len_1fps(total_frames: int, fps: float) -> int:
    """Convert raw (num_frames, fps) into a 1fps timeline length (seconds)."""
    if int(total_frames or 0) <= 0:
        return 0
    fps = float(fps or 0.0)
    if fps <= 0:
        fps = 30.0
    duration_s = float(total_frames) / fps
    return max(1, int(math.ceil(duration_s)))


def _timeline_to_frame_idx(timeline_idx: int, fps: float, total_frames: int) -> int:
    """Map 1fps timeline index (seconds) -> raw frame index for decoding."""
    total_frames = int(total_frames or 0)
    if total_frames <= 0:
        return 0
    fps = float(fps or 0.0)
    if fps <= 0:
        fps = 30.0
    t = max(0.0, float(timeline_idx))
    idx = int(t * fps)
    return max(0, min(idx, total_frames - 1))


def _extract_frames(video_path: str, frame_indices: list[int]) -> tuple[list[Image.Image], Optional[float]]:
    if not frame_indices:
        return [], None

    # Try decord first
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        images = [Image.fromarray(frame) for frame in frames]
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else None
        return images, fps
    except Exception:
        pass

    # Fall back to imageio
    try:
        import imageio

        reader = imageio.get_reader(video_path, "ffmpeg")
        images = [Image.fromarray(reader.get_data(idx)) for idx in frame_indices]
        meta = reader.get_meta_data()
        fps = meta.get("fps")
        reader.close()
        return images, fps
    except Exception as exc:
        raise RuntimeError(f"Failed to extract frames from {video_path}: {exc}")


def _format_question_block(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if labels:
        lines.append(
            "To answer, output <think>...</think> then <answer>LETTER</answer> "
            f"(LETTER must be one of: {', '.join(labels)})."
        )
    return "\n".join(lines)


def _build_user_content(
    question_block: str,
    summary: str,
    frame_count: int,
    round_idx: int,
    frame_indices: list[int],
    timestamps: list[Optional[float]],
    seen_frames: list[int],
    *,
    hide_seen_frames: bool = False,
    candidate_unseen_frames: Optional[list[int]] = None,
    use_candidate_frame_ids: bool = False,
    require_candidate_frames: bool = False,
    use_1fps_timeline: bool = False,
    time_reference: str = "",
) -> str:
    if question_block:
        header = f"Round {round_idx} / Question:\n{question_block}"
    else:
        header = f"Round {round_idx} (same question/options as Round 1)."

    if use_1fps_timeline:
        lines: list[str] = [header, f"Total frames L = {frame_count} (1 fps timeline)."]
    else:
        lines = [header, f"Total frames L = {frame_count}."]
    if time_reference:
        lines.append(f"Relevant time window for this question: {time_reference} (focus on this segment).")
    if hide_seen_frames:
        lines.append(
            f"Seen frames: {len(seen_frames)} frames already viewed "
            "(do NOT request any previously shown frames; follow the selection constraints below)."
        )
    else:
        lines.append(f"Seen frames (already viewed; do NOT request these again): {_format_frame_list(seen_frames)}")

    if candidate_unseen_frames and use_candidate_frame_ids:
        lines.append(
            "Candidate unseen frames available as IDs (all NEW): "
            f"choose IDs in [1, {len(candidate_unseen_frames)}]."
        )
        lines.append(
            "In <select>, output ONLY candidate IDs (comma-separated). Do NOT output raw frame indices when IDs exist."
        )
    else:
        lines.append(
            "Allowed unseen frame ranges for <select> (choose NEW indices only from these ranges): "
            f"{_format_intervals(_unseen_intervals(frame_count, seen_frames))}"
        )
        if candidate_unseen_frames:
            prefix = (
                "Candidate unseen frame ranges to request (REQUIRED, all NEW): "
                if require_candidate_frames
                else "Candidate unseen frame ranges to request (optional, all NEW): "
            )
            lines.append(prefix + f"{_format_intervals(_indices_to_intervals(candidate_unseen_frames))}")
            if require_candidate_frames:
                lines.append("In <select>, output ONLY indices within the Candidate unseen frame ranges above.")
    lines.extend(["Current summary:", f"<summarize>{summary}</summarize>", "Frames shown in this round:"])
    if hide_seen_frames or (candidate_unseen_frames and use_candidate_frame_ids):
        # Avoid leaking/copying raw frame indices when the action space is candidate IDs.
        for i, _ in enumerate(frame_indices):
            label = chr(ord("A") + i)
            lines.append(f"Shown frame {label} <image>")
    else:
        for idx, ts in zip(frame_indices, timestamps, strict=False):
            if ts is not None:
                lines.append(f"Frame {idx} (t={ts:.2f}s) <image>")
            else:
                lines.append(f"Frame {idx} <image>")
    return "\n".join(lines)


def _maybe_log_sample(payload: dict[str, Any]) -> None:
    """Optionally log a REVISE sample to disk for debugging."""
    log_dir = os.getenv("REVISE_LOG_DIR")
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "revise_samples.jsonl")

    def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Replace image blobs with tokens and keep text segments
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "image":
                        parts.append("<image>")
                    elif isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    else:
                        parts.append(str(c))
                content = "\n".join(parts)
            cleaned.append({**msg, "content": content})
        return cleaned

    safe_payload = payload.copy()
    if "messages" in safe_payload:
        safe_payload["messages"] = _strip_images(safe_payload["messages"])

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")


@register("revise_agent")
class ReviseAgentLoop(AgentLoopBase):
    """Multi-round controller implementing REVISE-style frame selection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.prompt_length = cfg.actor_rollout_ref.rollout.prompt_length
        self.response_length = cfg.actor_rollout_ref.rollout.response_length
        # NOTE: response_length is a padding/trajectory budget; generation is capped separately.
        # OmegaConf may contain explicit nulls; treat None as "unset".
        max_new_tokens = cfg.actor_rollout_ref.rollout.get("max_new_tokens", None)
        if max_new_tokens is None:
            max_new_tokens = self.response_length
        self.max_new_tokens = int(max_new_tokens)

        max_model_len = cfg.actor_rollout_ref.rollout.get("max_model_len", None)
        if max_model_len is None:
            max_model_len = self.prompt_length
        self.max_model_len = int(max_model_len)
        revise_cfg = cfg.actor_rollout_ref.rollout.get("revise", {})

        self.max_rounds = int(revise_cfg.get("max_rounds", 4))
        self.max_frames_per_round = int(revise_cfg.get("max_frames_per_round", 3))
        # Cap the total number of vision inputs carried across multi-round context.
        self.max_vision_inputs = int(revise_cfg.get("max_vision_inputs", 2))
        self.max_retries = int(revise_cfg.get("max_retries_per_round", 0))
        self.terminate_on_invalid_action = bool(revise_cfg.get("terminate_on_invalid_action", True))
        self.initial_sampling = revise_cfg.get("initial_sampling", "uniform")
        self.include_timestamps = bool(revise_cfg.get("include_timestamps", True))
        self.system_prompt_template = revise_cfg.get("system_prompt_template", DEFAULT_SYSTEM_PROMPT)
        self.hide_seen_frames_in_prompt = bool(revise_cfg.get("hide_seen_frames_in_prompt", False))
        self.include_candidate_frames_in_prompt = bool(revise_cfg.get("include_candidate_frames_in_prompt", False))
        self.candidate_frames_in_prompt_k = int(revise_cfg.get("candidate_frames_in_prompt_k", 12))
        self.use_candidate_frame_ids = bool(revise_cfg.get("use_candidate_frame_ids", False))
        self.require_candidate_frames = bool(revise_cfg.get("require_candidate_frames", False))
        self.use_1fps_timeline = bool(revise_cfg.get("use_1fps_timeline", False))
        self.seed = int(revise_cfg.get("seed", 0))

        # Optional EAGER-style margin scoring (for dense, annotation-free rewards).
        self.compute_margins = bool(revise_cfg.get("compute_margins", False))
        self.margin_logprobs_k = int(revise_cfg.get("margin_logprobs_k", 64))
        self.margin_temperature = float(revise_cfg.get("margin_temperature", 1.0))
        self.summary_only_logprobs_k = int(revise_cfg.get("summary_only_logprobs_k", self.margin_logprobs_k))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {})
        reward_model = kwargs.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", {})
        question = extra_info.get("question", "")
        choices = extra_info.get("choices", [])
        video_path = extra_info.get("video_path")
        frame_count = _safe_nonnegative_int(extra_info.get("frame_count", 0)) or 0
        time_reference = str(extra_info.get("time_reference") or "").strip()

        if not video_path:
            raise ValueError("extra_info.video_path is required for ReviseAgentLoop")

        total_frames: Optional[int] = None
        video_fps: Optional[float] = None

        if self.use_1fps_timeline:
            # Action indices are seconds on a 1-fps timeline; map to raw video frames for decoding.
            total_frames, video_fps = _load_video_meta(video_path)
            total_frames = _safe_nonnegative_int(total_frames) or 0
            video_fps = _safe_positive_float(video_fps) or 0.0
            if video_fps <= 0:
                video_fps = 30.0
            if frame_count <= 0:
                frame_count = _timeline_len_1fps(int(total_frames or 0), float(video_fps or 0.0))
        else:
            # If frame count missing, try to load from video metadata.
            if frame_count <= 0:
                frame_count, video_fps = _load_video_meta(video_path)
                frame_count = _safe_nonnegative_int(frame_count) or 0

        def _extract_action_frames(indices: list[int]) -> tuple[list[Image.Image], Optional[float]]:
            if not indices:
                return [], video_fps
            if not self.use_1fps_timeline:
                return _extract_frames(video_path, indices)
            tf = int(total_frames or 0)
            fps = float(video_fps or 0.0)
            if tf <= 0:
                return _extract_frames(video_path, indices)
            mapped = [_timeline_to_frame_idx(i, fps, tf) for i in indices]
            images, _ = _extract_frames(video_path, mapped)
            return images, fps

        rng = random.Random(self.seed)

        # Initial summary state (avoid placeholder tokens like "unknown"/"none" to reduce copying).
        summary_state = (
            "P: the agent has not seen any frames yet; "
            "O: no reliable observation yet; "
            "H: my belief will be updated based on what is observed; "
            "U: key detail is still unclear; "
            "R: need evidence from frames"
        )

        # Sample initial frames
        if self.initial_sampling == "random" and frame_count > 0:
            init_indices = rng.sample(range(frame_count), k=min(self.max_frames_per_round, frame_count))
        else:
            init_indices = _sample_uniform_indices(frame_count, self.max_frames_per_round)

        seen_frames = []
        all_images: list[Image.Image] = []

        init_frames = [idx for idx in init_indices if idx >= 0]
        init_images, fps = _extract_action_frames(init_frames)
        if len(init_images) > self.max_vision_inputs:
            init_images = init_images[: self.max_vision_inputs]
            init_frames = init_frames[: len(init_images)]
        timestamps = []
        for idx in init_frames:
            if not self.include_timestamps:
                timestamps.append(None)
                continue
            if self.use_1fps_timeline:
                timestamps.append(float(idx))
            elif fps:
                timestamps.append(idx / fps)
            else:
                timestamps.append(None)

        all_images.extend(init_images)
        seen_frames.extend(init_frames)

        question_block = _format_question_block(question, choices)
        system_prompt = self.system_prompt_template.format(max_frames_per_round=self.max_frames_per_round)

        candidate_unseen = None
        if self.include_candidate_frames_in_prompt:
            candidate_unseen = _propose_candidate_unseen_frames(
                frame_count=frame_count,
                seen=set(seen_frames),
                k=max(1, self.candidate_frames_in_prompt_k),
                rng=rng,
            )
        user_content = _build_user_content(
            question_block,
            summary_state,
            frame_count,
            round_idx=1,
            frame_indices=init_frames,
            timestamps=timestamps,
            seen_frames=seen_frames,
            hide_seen_frames=self.hide_seen_frames_in_prompt,
            candidate_unseen_frames=candidate_unseen,
            use_candidate_frame_ids=self.use_candidate_frame_ids,
            require_candidate_frames=self.require_candidate_frames,
            use_1fps_timeline=self.use_1fps_timeline,
            time_reference=time_reference,
        )

        def _with_images(message_list, images):
            """Convert plain-text message content to multimodal lists containing the
            provided images followed by the text. Needed for Qwen2.5-VL so that
            the processor can emit vision placeholders matching `image_data`.

            Only attach images to the final (typically user) message to avoid
            duplicating placeholders across system/assistant turns."""
            if not images:
                return message_list
            new_msgs = []
            for idx, m in enumerate(message_list):
                content = m["content"]
                # Only the last message gets the images.
                if idx != len(message_list) - 1 or isinstance(content, list):
                    new_msgs.append(m)
                    continue
                content_list = [{"type": "image", "image": img} for img in images]
                content_list.append({"type": "text", "text": content})
                new_msgs.append({**m, "content": content_list})
            return new_msgs

        messages = _with_images(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            all_images,
        )

        prompt_ids = await self.apply_chat_template(messages, images=all_images)
        # Guard against rare cases where vision tokens make the prompt exceed vLLM's
        # max_model_len (e.g., very large frames). Fall back to fewer initial frames.
        # Keep some extra headroom beyond max_new_tokens to avoid max_possible_tokens=0 edge cases.
        min_generation_room = int(self.max_new_tokens) + 32
        max_prompt_len = max(0, self.max_model_len - min_generation_room)
        while len(prompt_ids) > max_prompt_len and all_images:
            all_images = all_images[:-1]
            init_frames = init_frames[: len(all_images)]
            seen_frames = seen_frames[: len(all_images)]
            timestamps = timestamps[: len(all_images)]

            candidate_unseen = None
            if self.include_candidate_frames_in_prompt:
                candidate_unseen = _propose_candidate_unseen_frames(
                    frame_count=frame_count,
                    seen=set(seen_frames),
                    k=max(1, self.candidate_frames_in_prompt_k),
                    rng=rng,
                )
            user_content = _build_user_content(
                question_block,
                summary_state,
                frame_count,
                round_idx=1,
                frame_indices=init_frames,
                timestamps=timestamps,
                seen_frames=seen_frames,
                hide_seen_frames=self.hide_seen_frames_in_prompt,
                candidate_unseen_frames=candidate_unseen,
                use_candidate_frame_ids=self.use_candidate_frame_ids,
                require_candidate_frames=self.require_candidate_frames,
                use_1fps_timeline=self.use_1fps_timeline,
                time_reference=time_reference,
            )
            messages = _with_images(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                all_images,
            )
            prompt_ids = await self.apply_chat_template(messages, images=all_images)
        response_mask: list[int] = []
        response_logprobs: list[float] = []

        num_rounds = 0
        answer_text: Optional[str] = None
        format_valid = False
        last_response_text = ""
        invalid_outputs = 0
        invalid_attempts = 0
        total_retries = 0
        frames_all_seen = 0
        effective_rounds = 0
        terminated_reason: Optional[str] = None
        terminated_invalid_action = 0

        # EAGER-style scoring signals (optional; only populated when enabled via config).
        margins: list[float] = []
        margin_pred_letters: list[str] = []
        actions: list[str] = []
        format_by_round: list[float] = []
        summary_only_pred_letter: Optional[str] = None
        summary_only_correct = 0.0

        retries_left = [self.max_retries]

        illegal_reasons = {
            # Action contract violations (should never yield progress/new frames).
            "missing_frames_tag",
            "empty_frames",
            "frames_all_seen",
            "too_many_frames",
            "frames_out_of_range",
            "frames_not_in_candidates",
            # Summary is required for both actions.
            "invalid_select_summary",
            "invalid_answer_summary",
            # Hard protocol violations.
            "invalid_think",
            "invalid_answer_letter",
        }

        async def _handle_invalid(reason: str, feedback: str, *, force_answer: bool = False) -> bool:
            nonlocal invalid_outputs, invalid_attempts, total_retries
            nonlocal terminated_reason, terminated_invalid_action
            invalid_outputs += 1
            invalid_attempts += 1
            terminated_reason = reason
            if self.terminate_on_invalid_action and reason in illegal_reasons:
                terminated_invalid_action = 1
                return False
            ok = await self._retry_with_feedback(
                feedback,
                messages,
                prompt_ids,
                response_mask,
                response_logprobs,
                retries_left,
                force_answer=force_answer,
            )
            if ok:
                total_retries += 1
            return ok

        def _option_letters(num: int) -> list[str]:
            n = int(num or 0)
            if n <= 0:
                n = 5
            return [chr(ord("A") + i) for i in range(n)]

        def _letter_token_id(letter: str) -> Optional[int]:
            ids = self.tokenizer.encode(str(letter), add_special_tokens=False)
            if len(ids) != 1:
                return None
            return int(ids[0])

        option_letters = _option_letters(len(choices))
        option_token_ids: dict[str, int] = {}
        for l in option_letters:
            tid = _letter_token_id(l)
            if tid is not None:
                option_token_ids[l] = tid

        answer_idx = ground_truth.get("answer_idx") if isinstance(ground_truth, dict) else None
        correct_letter: Optional[str] = None
        try:
            if answer_idx is not None:
                correct_letter = chr(ord("A") + int(answer_idx))
        except Exception:
            correct_letter = None

        SCORE_SYSTEM_PROMPT = (
            "You are a multiple-choice video QA classifier. "
            "Given the question, options, current summary, and video frames, "
            "output ONLY the single best option letter (A/B/C/D/E)."
        )
        SCORE_SUMMARY_ONLY_SYSTEM_PROMPT = (
            "You are a multiple-choice QA classifier. "
            "Given the question, options, and the summary (no video frames), "
            "output ONLY the single best option letter (A/B/C/D/E)."
        )

        async def _score_letter_logprobs(
            *,
            system_prompt: str,
            user_text: str,
            images: Optional[list[Image.Image]],
            logprobs_k: int,
            temperature: float,
        ) -> Optional[dict[int, float]]:
            if not option_token_ids:
                return None
            score_messages = _with_images(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                images or [],
            )
            score_prompt_ids = await self.apply_chat_template(score_messages, images=images if images else None)
            if len(score_prompt_ids) >= self.max_model_len:
                return None
            try:
                out = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=score_prompt_ids,
                    sampling_params={
                        "max_tokens": 1,
                        "temperature": float(temperature),
                        "top_p": 1.0,
                        "logprobs": int(max(0, logprobs_k)),
                    },
                    image_data=images if images else None,
                    video_data=None,
                )
            except Exception:
                return None
            if not out.top_logprobs:
                return None
            return out.top_logprobs[0] or {}

        def _margin_from_logprobs(lp: dict[int, float]) -> tuple[float, str]:
            """Return (margin, argmax_letter) for the current state."""
            # Gather letter logprobs; fall back to a very small value when absent.
            letter_lps: dict[str, float] = {}
            for l, tid in option_token_ids.items():
                letter_lps[l] = float(lp.get(int(tid), -1e9))
            # Argmax among letters.
            pred = max(letter_lps.items(), key=lambda kv: kv[1])[0] if letter_lps else "A"
            if correct_letter is None or correct_letter not in letter_lps:
                return 0.0, pred
            correct_lp = letter_lps[correct_letter]
            best_other = max((v for k, v in letter_lps.items() if k != correct_letter), default=-1e9)
            return float(correct_lp - best_other), pred

        for round_idx in range(1, self.max_rounds + 1):
            num_rounds = round_idx

            if len(prompt_ids) >= self.max_model_len:
                logger.warning(
                    "Prompt already at/over max_model_len (%s >= %s); stopping sample early.",
                    len(prompt_ids),
                    self.max_model_len,
                )
                break

            # Compute EAGER margin m_t for the current state (before acting).
            if self.compute_margins and correct_letter is not None:
                score_user_text = (
                    f"{question_block}\n\n"
                    f"Current summary:\n{summary_state}\n\n"
                    f"Output ONLY one letter from: {', '.join(option_letters)}."
                )
                lp = await _score_letter_logprobs(
                    system_prompt=SCORE_SYSTEM_PROMPT,
                    user_text=score_user_text,
                    images=all_images,
                    logprobs_k=self.margin_logprobs_k,
                    temperature=self.margin_temperature,
                )
                if lp is None:
                    margins.append(0.0)
                    margin_pred_letters.append("")
                else:
                    m_t, pred = _margin_from_logprobs(lp)
                    margins.append(float(m_t))
                    margin_pred_letters.append(pred)

            with simple_timer("generate_sequences", {}):
                try:
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=all_images,
                        video_data=None,
                    )
                except Exception as exc:
                    logger.warning("vLLM generation failed (%s); stopping sample early.", exc)
                    break

            # Update response tracking
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs

            last_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": last_response_text})

            # Parse model output
            answer = _extract_tag(last_response_text, _ANSWER_RE)
            frames_text = _extract_tag(last_response_text, _SELECT_RE)
            summary = _extract_tag(last_response_text, _SUMMARIZE_RE)

            # Every response MUST begin with a <think> reasoning trace (paper protocol).
            think = _extract_tag(last_response_text, _THINK_RE)
            if think is None:
                feedback = (
                    "Invalid response: every response MUST begin with a <think>...</think> reasoning trace, "
                    "then either <summarize> + <select> (request) or <answer> (final)."
                )
                if not await _handle_invalid("missing_think", feedback):
                    break
                continue

            if answer:
                # Strict-paper Answer round: <think> + <answer> only. The last committed
                # <summarize> is reused as the persistent state, so no summary is required here.
                normalized = _normalize_answer_letter(answer, len(choices))
                if normalized is None:
                    allowed = [chr(ord("A") + i) for i in range(len(choices) or 5)]
                    feedback = (
                        "Invalid response: <answer> must be exactly ONE option letter "
                        f"({', '.join(allowed)}). Do not output words or a sentence."
                    )
                    if not await _handle_invalid("invalid_answer_letter", feedback, force_answer=True):
                        break
                    continue
                if summary is not None:
                    if not _is_placeholder(summary) and _summary_has_ohrpu(summary):
                        summary_state = summary
                answer_text = normalized
                format_valid = True
                actions.append("answer")
                format_by_round.append(1.0)
                break

            if frames_text is None:
                # invalid: missing frames for select
                feedback = (
                    "Invalid response: missing <select> tag for requesting more frames. "
                    "Remember: <select> must list NEW frame indices to view NEXT (not already seen)."
                )
                if not await _handle_invalid("missing_frames_tag", feedback):
                    break
                continue

            # Validate summary (required to proceed with frame selection).
            if summary is None or _is_placeholder(summary) or _contains_banned_example(summary) or not _summary_has_ohrpu(summary):
                feedback = (
                    "Invalid response: when requesting more frames, include a meaningful <summarize> with P/O/H/U/R "
                    "in that exact order (no placeholders like '.../none/unknown')."
                )
                if not await _handle_invalid("invalid_select_summary", feedback):
                    break
                continue

            summary_state = summary

            requested = _dedupe_preserve_order(_parse_frame_indices(frames_text))
            if self.use_candidate_frame_ids and candidate_unseen:
                mapped: list[int] = []
                invalid_id = False
                for cid in requested:
                    if 1 <= cid <= len(candidate_unseen):
                        mapped.append(int(candidate_unseen[cid - 1]))
                    else:
                        invalid_id = True
                if invalid_id:
                    feedback = (
                        "Invalid response: when Candidate Frame IDs are provided, "
                        f"output IDs in [1, {len(candidate_unseen)}] inside <select>."
                    )
                    if not await _handle_invalid("frames_out_of_range", feedback):
                        break
                    continue
                requested = _dedupe_preserve_order(mapped)
            elif self.require_candidate_frames and candidate_unseen:
                allowed = {int(i) for i in candidate_unseen}
                disallowed = [i for i in requested if int(i) not in allowed]
                if disallowed:
                    feedback = (
                        "Invalid response: when Candidate unseen frame ranges are provided, "
                        "request ONLY indices within those candidate ranges. "
                        "Candidate ranges: "
                        f"{_format_intervals(_indices_to_intervals(candidate_unseen))}."
                    )
                    if not await _handle_invalid("frames_not_in_candidates", feedback):
                        break
                    continue
            if not requested:
                feedback = (
                    "Invalid response: <select> is empty. "
                    "Provide 1–{max_k} NEW frame indices to view NEXT (comma-separated). "
                    "Do not request frames already in Seen frames."
                ).format(max_k=self.max_frames_per_round)
                if not await _handle_invalid("empty_frames", feedback):
                    break
                continue

            # Enforce constraints
            requested_unique = [i for i in requested if i not in seen_frames]
            if len(requested_unique) == 0:
                frames_all_seen += 1
                candidate_text = (
                    " Allowed unseen ranges: "
                    f"{_format_intervals(_unseen_intervals(frame_count, seen_frames))}."
                )
                feedback = (
                    "Invalid response: all requested frames are already seen. "
                    "In <select>, output 1–{max_k} NEW indices NOT in the Seen frames list."
                    + candidate_text
                ).format(max_k=self.max_frames_per_round)
                if not await _handle_invalid("frames_all_seen", feedback):
                    break
                continue
            if len(requested_unique) > self.max_frames_per_round:
                feedback = (
                    f"Invalid response: requested {len(requested_unique)} NEW frames, "
                    f"which exceeds max {self.max_frames_per_round}. "
                    f"Choose at most {self.max_frames_per_round} NEW frames NOT in Seen frames."
                )
                if not await _handle_invalid("too_many_frames", feedback):
                    break
                continue

            # Clip to range
            valid_requested = [i for i in requested_unique if 0 <= i < frame_count]
            if not valid_requested:
                feedback = (
                    "Invalid response: frame indices out of range. "
                    f"Valid range is [0, {max(0, frame_count - 1)}]. "
                    "Choose NEW frame indices within range and NOT in Seen frames."
                )
                if not await _handle_invalid("frames_out_of_range", feedback):
                    break
                continue

            # Enforce a global cap on total images carried across rounds.
            slots_left = self.max_vision_inputs - len(all_images)
            if slots_left <= 0:
                feedback = (
                    f"Vision limit of {self.max_vision_inputs} images reached. "
                    "Provide the final answer using <think>...</think> then <answer>LETTER</answer>."
                )
                await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                    force_answer=True,
                )
                continue

            if len(valid_requested) > slots_left:
                valid_requested = valid_requested[:slots_left]

            summary_state = summary
            candidate_images, fps = _extract_action_frames(valid_requested)
            candidate_timestamps: list[Optional[float]] = []
            for idx in valid_requested:
                if not self.include_timestamps:
                    candidate_timestamps.append(None)
                    continue
                if self.use_1fps_timeline:
                    candidate_timestamps.append(float(idx))
                elif fps:
                    candidate_timestamps.append(idx / fps)
                else:
                    candidate_timestamps.append(None)

            # Guard against vLLM context overflow (prompt must be <= max_model_len).
            # Images can contribute many tokens; for rare near-limit cases, reduce the
            # number of frames added for the next round and leave headroom to answer.
            min_generation_room = int(self.max_new_tokens) + 32
            max_prompt_len = max(0, self.max_model_len - min_generation_room)

            selected_frames: list[int] = []
            selected_images: list[Image.Image] = []
            selected_user_ids: Optional[list[int]] = None
            selected_messages: Optional[list[dict[str, Any]]] = None
            selected_candidate_unseen: Optional[list[int]] = None

            for k in range(len(valid_requested), 0, -1):
                trial_frames = valid_requested[:k]
                trial_images = candidate_images[:k]
                trial_timestamps = candidate_timestamps[:k]
                trial_seen = seen_frames + trial_frames
                trial_candidate_unseen = None
                if self.include_candidate_frames_in_prompt:
                    trial_candidate_unseen = _propose_candidate_unseen_frames(
                        frame_count=frame_count,
                        seen=set(trial_seen),
                        k=max(1, self.candidate_frames_in_prompt_k),
                        rng=rng,
                    )

                trial_user_content = _build_user_content(
                    question_block,
                    summary_state,
                    frame_count,
                    round_idx=round_idx + 1,
                    frame_indices=trial_frames,
                    timestamps=trial_timestamps,
                    seen_frames=trial_seen,
                    hide_seen_frames=self.hide_seen_frames_in_prompt,
                    candidate_unseen_frames=trial_candidate_unseen,
                    use_candidate_frame_ids=self.use_candidate_frame_ids,
                    require_candidate_frames=self.require_candidate_frames,
                    use_1fps_timeline=self.use_1fps_timeline,
                    time_reference=time_reference,
                )
                trial_messages = _with_images(
                    [{"role": "user", "content": trial_user_content}],
                    trial_images,
                )
                trial_user_ids = await self.apply_chat_template(
                    trial_messages,
                    images=trial_images,
                    remove_system_prompt=True,
                )

                # Also keep trajectory within the configured response_length budget to avoid
                # truncating inside a vision token block (which can break Qwen2.5-VL RoPE indexing).
                response_budget_ok = (
                    len(response_mask) + len(trial_user_ids) + int(self.max_new_tokens) + 16
                    <= int(self.response_length)
                )

                if len(prompt_ids) + len(trial_user_ids) <= max_prompt_len and response_budget_ok:
                    selected_frames = trial_frames
                    selected_images = trial_images
                    selected_user_ids = trial_user_ids
                    selected_messages = trial_messages
                    selected_candidate_unseen = trial_candidate_unseen
                    break

            if selected_user_ids is None or selected_messages is None:
                feedback = (
                    "Context/trajectory length limit reached. Provide the final answer now using "
                    "<think>...</think> then <answer>LETTER</answer>."
                )
                await self._retry_with_feedback(
                    feedback,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs,
                    retries_left,
                    force_answer=True,
                )
                continue

            all_images.extend(selected_images)
            seen_frames.extend(selected_frames)
            messages.extend(selected_messages)
            prompt_ids += selected_user_ids
            response_mask += [0] * len(selected_user_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(selected_user_ids)
            candidate_unseen = selected_candidate_unseen
            effective_rounds += 1
            actions.append("select")
            format_by_round.append(1.0)

        # If no answer was produced, force a final answer attempt
        if answer_text is None and terminated_invalid_action == 0:
            feedback = (
                "Max rounds reached. Provide final answer now using <think>...</think> then <answer>LETTER</answer>."
            )
            await self._retry_with_feedback(
                feedback,
                messages,
                prompt_ids,
                response_mask,
                response_logprobs,
                retries_left,
                force_answer=True,
            )

            # We won't enter another normal round to score m_t for the final state; compute it here
            # so the last Select can still receive a confidence-gain signal.
            if self.compute_margins and correct_letter is not None:
                score_user_text = (
                    f"{question_block}\n\n"
                    f"Current summary:\n{summary_state}\n\n"
                    f"Output ONLY one letter from: {', '.join(option_letters)}."
                )
                lp = await _score_letter_logprobs(
                    system_prompt=SCORE_SYSTEM_PROMPT,
                    user_text=score_user_text,
                    images=all_images,
                    logprobs_k=self.margin_logprobs_k,
                    temperature=self.margin_temperature,
                )
                if lp is None:
                    margins.append(0.0)
                    margin_pred_letters.append("")
                else:
                    m_t, pred = _margin_from_logprobs(lp)
                    margins.append(float(m_t))
                    margin_pred_letters.append(pred)

            if len(prompt_ids) < self.max_model_len:
                try:
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=all_images,
                        video_data=None,
                    )
                except Exception as exc:
                    logger.warning("vLLM final-answer generation failed (%s); leaving answer empty.", exc)
                    output = None
            else:
                output = None

            if output is not None:
                response_ids = output.token_ids
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)
                if output.log_probs:
                    response_logprobs += output.log_probs
                last_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": last_response_text})
                forced_answer = _extract_tag(last_response_text, _ANSWER_RE)
                answer_text = _normalize_answer_letter(forced_answer or "", len(choices))
                forced_summary = _extract_tag(last_response_text, _SUMMARIZE_RE)
                if forced_summary is not None:
                    summary_state = forced_summary
                format_valid = answer_text is not None
                actions.append("answer")
                format_by_round.append(1.0 if format_valid else 0.0)

        # Summary-only sufficiency signal for EAGER (evaluate answerability from summary alone).
        if self.compute_margins and correct_letter is not None and answer_text is not None:
            summary_user_text = (
                f"{question_block}\n\n"
                f"Summary:\n{summary_state}\n\n"
                f"Output ONLY one letter from: {', '.join(option_letters)}."
            )
            lp = await _score_letter_logprobs(
                system_prompt=SCORE_SUMMARY_ONLY_SYSTEM_PROMPT,
                user_text=summary_user_text,
                images=None,
                logprobs_k=self.summary_only_logprobs_k,
                temperature=self.margin_temperature,
            )
            if lp is not None:
                _, pred = _margin_from_logprobs(lp)
                summary_only_pred_letter = pred
                summary_only_correct = 1.0 if pred == correct_letter else 0.0

        # Prepare output
        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_only_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        revise_metrics = {
            "num_rounds": num_rounds,
            "effective_rounds": effective_rounds,
            "frames_used": len(seen_frames),
            "seen_frames": seen_frames,
            "summary": summary_state,
            "answer": answer_text,
            "format_valid": format_valid,
            "last_response": last_response_text,
            "invalid_outputs": invalid_outputs,
            "invalid_attempts": invalid_attempts,
            "total_retries": total_retries,
            "frames_all_seen": frames_all_seen,
            "terminated_reason": terminated_reason,
            "illegal_action": terminated_invalid_action,
            # Optional EAGER signals (only populated when compute_margins=True).
            "margins": margins,
            "margin_pred_letters": margin_pred_letters,
            "actions": actions,
            "format_by_round": format_by_round,
            "summary_only_pred": summary_only_pred_letter,
            "summary_only_correct": summary_only_correct,
            "margin_logprobs_k": self.margin_logprobs_k,
            "margin_temperature": self.margin_temperature,
        }

        # Optional: summary-only correctness approximation
        if ground_truth and summary_state:
            gt_text = ground_truth.get("answer_text")
            if gt_text:
                revise_metrics["summary_contains_answer"] = str(gt_text).lower() in summary_state.lower()

        _maybe_log_sample(
            {
                "timestamp": time.time(),
                "sample_id": extra_info.get("sample_id"),
                "video_id": extra_info.get("video_id"),
                "video_path": video_path,
                "question": question,
                "choices": choices,
                "ground_truth": ground_truth,
                "messages": messages,
                "seen_frames": seen_frames,
                "num_rounds": num_rounds,
                "summary": summary_state,
                "answer": answer_text,
                "format_valid": format_valid,
                "revise_metrics": revise_metrics,
            }
        )

        output = AgentLoopOutput(
            prompt_ids=prompt_only_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={"images": all_images},
            num_turns=1 + 2 * num_rounds,
            metrics={},
            extra_fields={"revise": revise_metrics},
        )
        return output

    async def _retry_with_feedback(
        self,
        feedback: str,
        messages: list[dict[str, Any]],
        prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        retries_left: list[int],
        force_answer: bool = False,
    ) -> bool:
        """Append feedback and request a retry. Returns False if retry budget exceeded."""
        if retries_left[0] <= 0 and not force_answer:
            return False

        feedback_text = (
            f"{feedback}\n"
            "Please respond with one of the required formats."
        )
        if force_answer:
            feedback_text = (
                f"{feedback}\n"
                "Output ONLY <think>...</think> then <answer>LETTER</answer> (no <summarize> on the answer round). "
                "In <think>, briefly justify the choice. "
                "In <answer>, LETTER must be a single option letter (e.g., A/B/C/D/E)."
            )

        add_messages = [{"role": "user", "content": feedback_text}]
        messages.extend(add_messages)
        user_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        prompt_ids += user_ids
        response_mask += [0] * len(user_ids)
        if response_logprobs:
            response_logprobs += [0.0] * len(user_ids)

        if force_answer:
            return True

        retries_left[0] -= 1
        return True
