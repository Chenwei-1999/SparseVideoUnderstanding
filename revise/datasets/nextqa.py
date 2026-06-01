"""NExT-QA loading and adapter code for the shared REVISE PnP harness."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from PIL import Image

from revise.pnp.policy import build_retry_feedback, decide_revise_action, final_answer_instruction
from revise.pnp.prompts import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from revise.pnp.protocols import LoopConfig
from revise.pnp.utils import (
    ANSWER_RE,
    FORCE_ANSWER_INSTRUCTIONS_POHR,
    SELECT_RE,
    SUMMARIZE_RE,
    THINK_RE,
    build_revise_user_text,
    contains_banned_example,
    dedupe_preserve_order,
    default_initial_summary,
    extract_frames_1fps,
    extract_tag,
    extract_video_info,
    format_revise_question_block,
    in_intervals,
    is_placeholder,
    normalize_answer_letter,
    normalize_video_id,
    parse_int_list,
    propose_candidate_frames,
    resolve_nextqa_video_path,
    retry_feedback_text,
    sample_uniform_indices,
    stable_sample_id_nextqa,
    summary_has_ohrpu,
    summary_has_stale_boilerplate,
    timeline_len_1fps,
    unseen_intervals,
)

_CAPTION_CACHE: dict[tuple[str, str], dict[int, str]] = {}
_FPS_CACHE: dict[str, float] = {}
_TIMELINE_CACHE: dict[tuple[str, int], int] = {}


@dataclass
class NextQASample:
    sample_id: str
    qid: str
    video_id: str
    video_path: str
    question: str
    choices: list[str]
    answer_idx: int
    frame_count: int


def load_samples(
    csv_path: str,
    map_json: str,
    video_root: str,
    max_samples: int,
    seed: int,
) -> list[NextQASample]:
    with open(map_json, encoding="utf-8") as f:
        video_map = json.load(f)
    video_map = {str(k): v for k, v in video_map.items()}

    df = pd.read_csv(csv_path)
    if max_samples > 0:
        # Deterministic shuffle, but keep picking until we have enough valid samples.
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    samples: list[NextQASample] = []
    for _, row in df.iterrows():
        video_id = normalize_video_id(row["video"])
        rel = video_map.get(video_id)
        if rel is None:
            continue
        video_path = resolve_nextqa_video_path(video_root, str(rel), video_id)
        if video_path is None:
            continue
        choices = [str(row[f"a{i}"]) for i in range(5)]
        answer_idx = int(row.get("answer", 0))
        samples.append(
            NextQASample(
                sample_id=stable_sample_id_nextqa(video_id, str(row.get("question", "")), choices, answer_idx),
                qid=str(row.get("qid", "")),
                video_id=video_id,
                video_path=video_path,
                question=str(row.get("question", "")),
                choices=choices,
                answer_idx=answer_idx,
                frame_count=int(row.get("frame_count", 0)),
            )
        )
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def load_progress_from_log(
    log_path: str,
    max_rounds: int,
    default_num_choices: int = 5,
    sample_ids: set[str] | None = None,
) -> tuple[int, int, int, set[str]]:
    if not log_path or not os.path.exists(log_path):
        return 0, 0, 0, set()

    allowed_sample_ids = {str(sample_id) for sample_id in sample_ids or set() if sample_id}
    completed_ids: set[str] = set()
    correct = 0
    total_rounds = 0

    def _int_from_obj(obj: dict[str, Any], key: str, default: int) -> int:
        try:
            return int(obj.get(key, default))
        except Exception:
            return default

    def _accepted_answer_letter(obj: dict[str, Any], num_choices: int, round_idx: int) -> Optional[str]:
        if obj.get("action_kind") == "invalid" or obj.get("invalid_reason"):
            return None

        raw = str(obj.get("raw_output", ""))
        if not raw:
            return None
        seen_frames = obj.get("seen_frames")
        if not isinstance(seen_frames, list):
            seen_frames = []
        candidate_frames = obj.get("candidate_unseen_frames")
        if not isinstance(candidate_frames, list):
            candidate_frames = []
        frame_count = _int_from_obj(obj, "timeline_frame_count", _int_from_obj(obj, "raw_frame_count", 0))
        decision = decide_revise_action(
            raw,
            num_choices=num_choices,
            seen_frames=seen_frames,
            frame_count=max(1, frame_count),
            max_frames_per_round=_int_from_obj(obj, "max_frames_per_round", 10**9),
            round_idx=round_idx,
            max_rounds=max_rounds,
            answer_only_final_round=bool(obj.get("answer_only_final_round", False)),
            min_select_rounds=_int_from_obj(obj, "min_select_rounds", 0),
            select_rounds_so_far=max(0, round_idx - 1),
            candidate_frames=candidate_frames,
            use_candidate_frame_ids=bool(obj.get("use_candidate_frame_ids", False)),
            require_candidate_frames=bool(obj.get("require_candidate_frames", False)),
        )
        if decision.kind != "answer":
            return None
        accepted = obj.get("accepted_answer_letter") or obj.get("final_answer")
        if accepted and normalize_answer_letter(str(accepted), num_choices) != decision.answer_letter:
            return None
        return decision.answer_letter

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            sample_id = str(obj.get("sample_id", "") or "")
            if not sample_id or sample_id in completed_ids:
                continue
            if allowed_sample_ids and sample_id not in allowed_sample_ids:
                continue

            choices = obj.get("choices", [])
            num_choices = len(choices) if isinstance(choices, list) and choices else default_num_choices
            round_idx = _int_from_obj(obj, "round_idx", 0)
            answer_letter = _accepted_answer_letter(obj, num_choices, round_idx)
            if answer_letter is None:
                continue

            completed_ids.add(sample_id)
            pred_idx = ord(answer_letter) - ord("A")
            if _int_from_obj(obj, "ground_truth_idx", -1) == pred_idx:
                correct += 1
            total_rounds += min(round_idx, max_rounds)

    return len(completed_ids), correct, total_rounds, completed_ids


def _load_video_captions(captions_dir: str, video_id: str) -> dict[int, str]:
    cache_key = (str(captions_dir), str(video_id))
    cached = _CAPTION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    path = os.path.join(captions_dir, f"{video_id}_cap.json")
    if not os.path.exists(path):
        _CAPTION_CACHE[cache_key] = {}
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        _CAPTION_CACHE[cache_key] = {}
        return {}
    captions: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if isinstance(value, str):
                captions[idx] = value.strip()
    _CAPTION_CACHE[cache_key] = captions
    return captions


def _get_video_fps(video_path: str) -> float:
    cached = _FPS_CACHE.get(video_path)
    if cached is not None:
        return cached
    fps = 0.0
    try:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 0.0
    _FPS_CACHE[video_path] = fps
    return fps


def _timeline_frame_count(video_path: str, fallback_raw_frame_count: int) -> int:
    """Return the NExT-QA action-space length on a 1-fps timeline."""

    fallback = max(0, int(fallback_raw_frame_count or 0))
    cache_key = (str(video_path), fallback)
    cached = _TIMELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    raw_frame_count, fps = extract_video_info(video_path)
    if raw_frame_count <= 0:
        raw_frame_count = fallback
    timeline_count = timeline_len_1fps(raw_frame_count, fps) if raw_frame_count > 0 else 0
    _TIMELINE_CACHE[cache_key] = int(timeline_count)
    return int(timeline_count)


def _frames_has_range_syntax(frames_text: str) -> bool:
    if not frames_text:
        return False
    # Common failure mode: model copies allowed ranges like "4-182" into <select>.
    return bool(re.search(r"\d+\s*[-–—]\s*\d+", frames_text))


def _format_question(question: str, choices: list[str]) -> str:
    return format_revise_question_block(question, choices)


def _build_user_text(
    question_block: str,
    summary: str,
    frame_count: int,
    round_idx: int,
    frame_indices: list[int],
    seen_frames: list[int],
    *,
    render_images: bool = True,
    hide_seen_frames: bool = False,
    candidate_unseen_frames: Optional[list[int]] = None,
    use_candidate_frame_ids: bool = False,
    require_candidate_frames: bool = False,
    shown_frame_captions: Optional[list[str]] = None,
    candidate_id_captions: Optional[list[str]] = None,
    shown_frame_ts: Optional[list[int]] = None,
    candidate_id_ts: Optional[list[int]] = None,
    timestamps: Optional[list[Optional[float]]] = None,
    use_1fps_timeline: bool = False,
) -> str:
    return build_revise_user_text(
        question_block=question_block,
        summary=summary,
        frame_count=frame_count,
        round_idx=round_idx,
        frame_indices=frame_indices,
        seen_frames=seen_frames,
        render_images=render_images,
        hide_seen_frames=hide_seen_frames,
        candidate_unseen_frames=candidate_unseen_frames,
        use_candidate_frame_ids=use_candidate_frame_ids,
        require_candidate_frames=require_candidate_frames,
        shown_frame_captions=shown_frame_captions,
        candidate_id_captions=candidate_id_captions,
        shown_frame_ts=shown_frame_ts,
        candidate_id_ts=candidate_id_ts,
        timestamps=timestamps,
        use_1fps_timeline=use_1fps_timeline,
    )


def _sample_unseen_frames(frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
    if frame_count <= 0 or k <= 0:
        return []
    if len(seen) >= frame_count:
        return []
    candidates = [i for i in range(frame_count) if i not in seen]
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates
    return sorted(rng.sample(candidates, k=k))


def _retry_feedback_text(feedback: str, *, force_answer: bool = False) -> str:
    """Compatibility wrapper for legacy NExT-QA launchers/tests."""

    return retry_feedback_text(
        feedback,
        force_answer=force_answer,
        force_instructions=FORCE_ANSWER_INSTRUCTIONS_POHR,
    )


class NextQADataset:
    def video_path(self, sample: NextQASample) -> str:
        return sample.video_path

    def frame_count(self, sample: NextQASample) -> int:
        return _timeline_frame_count(sample.video_path, sample.frame_count)

    def video_id(self, sample: NextQASample) -> str:
        return sample.video_id

    def num_choices(self, sample: NextQASample) -> int:
        return len(sample.choices)

    def normalize_answer(self, sample: NextQASample, answer_text: str) -> Optional[str]:
        return normalize_answer_letter(answer_text, self.num_choices(sample))

    def ground_truth_letter(self, sample: NextQASample) -> Optional[str]:
        if 0 <= int(sample.answer_idx) < self.num_choices(sample):
            return chr(ord("A") + int(sample.answer_idx))
        return None

    def is_correct(self, sample: NextQASample, pred_letter: str) -> bool:
        return pred_letter == self.ground_truth_letter(sample)

    def log_fields(self, sample: NextQASample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "qid": sample.qid,
            "video_id": sample.video_id,
            "video_path": sample.video_path,
            "raw_frame_count": sample.frame_count,
            "timeline_frame_count": self.frame_count(sample),
            "question": sample.question,
            "choices": sample.choices,
            "ground_truth_idx": sample.answer_idx,
        }

    def format_question(self, sample: NextQASample) -> str:
        return _format_question(sample.question, sample.choices)

    def system_prompt(self, cfg: LoopConfig) -> str:
        return DEFAULT_SYSTEM_PROMPT.format(max_frames_per_round=cfg.max_frames_per_round)

    def build_user_text(self, **kwargs: Any) -> str:
        kwargs.setdefault("use_1fps_timeline", True)
        kwargs.setdefault("timestamps", [float(i) for i in kwargs.get("frame_indices", [])])
        return _build_user_text(**kwargs)

    def extract_frames(self, sample: NextQASample, indices: list[int]) -> list[Image.Image]:
        return extract_frames_1fps(sample.video_path, indices)

    def sample_unseen_frames(self, frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
        return _sample_unseen_frames(frame_count, seen, k, rng=rng)

    def initial_frame_indices(self, sample: NextQASample, frame_count: int, cfg: LoopConfig) -> list[int]:
        _ = sample
        return sample_uniform_indices(frame_count, cfg.max_frames_per_round)

    def candidate_frame_indices(
        self,
        sample: NextQASample,
        *,
        frame_count: int,
        seen_frames: list[int],
        k: int,
        rng: random.Random,
    ) -> list[int]:
        _ = sample
        return propose_candidate_frames(frame_count=frame_count, seen=set(seen_frames), k=k, rng=rng)

    def fallback_frame_indices(self, sample: NextQASample, frame_count: int, k: int, cfg: LoopConfig) -> list[int]:
        _ = sample, cfg
        return sample_uniform_indices(frame_count, k)

    def retry_feedback_text(
        self,
        reason: str,
        *,
        force_answer: bool = False,
        max_frames_per_round: int = 0,
        frame_count: int = 0,
        seen_frames: Optional[list[int]] = None,
    ) -> str:
        return build_retry_feedback(
            reason,
            force_answer=force_answer,
            max_frames_per_round=max_frames_per_round,
            frame_count=frame_count,
            seen_frames=seen_frames,
            num_choices=5,
        )

    def load_video_captions(self, captions_dir: str, video_id: str) -> dict[int, str]:
        return _load_video_captions(captions_dir, video_id)

    def get_video_fps(self, video_path: str) -> float:
        return _get_video_fps(video_path)

    def caption_key_for_frame_index(self, frame_idx: int, fps: float) -> int:
        _ = fps
        return int(frame_idx)

    def parse_think(self, raw: str) -> Optional[str]:
        return extract_tag(raw, THINK_RE)

    def parse_summary(self, raw: str) -> Optional[str]:
        return extract_tag(raw, SUMMARIZE_RE)

    def parse_answer(self, raw: str) -> Optional[str]:
        return extract_tag(raw, ANSWER_RE)

    def parse_select(self, raw: str) -> Optional[str]:
        return extract_tag(raw, SELECT_RE)

    def should_commit_summary(self, summary: Optional[str], *, seen_count: int) -> bool:
        return (
            bool(summary)
            and (not is_placeholder(summary))
            and (not contains_banned_example(summary))
            and summary_has_ohrpu(summary)
            and (not summary_has_stale_boilerplate(summary, seen_count=seen_count))
        )

    def is_select_summary_valid(self, summary: Optional[str], *, seen_count: int) -> bool:
        _ = seen_count
        return (
            summary is not None
            and (not is_placeholder(summary))
            and (not contains_banned_example(summary))
            and summary_has_ohrpu(summary)
        )

    def is_summary_stale(self, summary: Optional[str], *, seen_count: int) -> bool:
        if summary is None:
            return False
        return summary_has_stale_boilerplate(summary, seen_count=seen_count)

    def select_has_range_syntax(self, frames_text: str) -> bool:
        return _frames_has_range_syntax(frames_text)

    def parse_select_frames(self, frames_text: str) -> list[int]:
        return dedupe_preserve_order(parse_int_list(frames_text))

    def map_candidate_frame_ids(
        self,
        requested_ids: list[int],
        candidate_frames: list[int],
    ) -> Optional[list[int]]:
        mapped: list[int] = []
        for cid in requested_ids:
            if 1 <= cid <= len(candidate_frames):
                mapped.append(int(candidate_frames[cid - 1]))
            else:
                return None
        return dedupe_preserve_order(mapped)

    def filter_requested_frames(
        self,
        requested_frames: list[int],
        *,
        frame_count: int,
        seen_frames: list[int],
        candidate_frames: list[int],
        require_candidate_frames: bool = False,
    ) -> tuple[list[int], Optional[str]]:
        if require_candidate_frames and candidate_frames:
            allowed = {int(i) for i in candidate_frames}
            disallowed = [i for i in requested_frames if int(i) not in allowed]
            if disallowed:
                return [], "frames_not_in_candidates"
            return (
                [
                    i
                    for i in requested_frames
                    if 0 <= i < frame_count and i not in seen_frames and int(i) in allowed
                ],
                None,
            )
        allowed_ranges = unseen_intervals(frame_count, seen_frames)
        return (
            [
                i
                for i in requested_frames
                if 0 <= i < frame_count and i not in seen_frames and in_intervals(i, allowed_ranges)
            ],
            None,
        )

    def initial_summary(self, cfg: LoopConfig) -> str:
        _ = cfg
        return default_initial_summary()

    def final_round_instruction(self, cfg: LoopConfig) -> Optional[str]:
        _ = cfg
        return final_answer_instruction()

    def final_answer_instruction(self, cfg: LoopConfig) -> Optional[str]:
        _ = cfg
        return final_answer_instruction()

    def forced_answer_request(
        self,
        sample: NextQASample,
        *,
        question_block: str,
        frame_count: int,
        max_rounds: int,
        system_prompt: str,
        last_user_text: str,
        last_images: list[Any],
    ) -> tuple[str, str, list[Any]]:
        _ = sample, question_block, frame_count, max_rounds
        return system_prompt, f"{last_user_text}\n\n{final_answer_instruction()}", last_images

    def should_terminate_on_invalid_summary(self, cfg: LoopConfig) -> bool:
        _ = cfg
        return False

    def should_fail_on_empty_images(self, cfg: LoopConfig) -> bool:
        _ = cfg
        return False

    def should_count_exhausted_invalid_as_retry(self, reason: str) -> bool:
        _ = reason
        return False

    def should_clear_frame_plan_on_exhausted_invalid(self, reason: str) -> bool:
        _ = reason
        return False

    def should_retry_invalid_output(self, reason: str) -> bool:
        _ = reason
        return True


_load_nextqa_samples = load_samples
_load_progress_from_log = load_progress_from_log

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "NextQADataset",
    "NextQASample",
    "_TIMELINE_CACHE",
    "_build_user_text",
    "_format_question",
    "_load_nextqa_samples",
    "_load_progress_from_log",
    "extract_frames_1fps",
    "extract_video_info",
    "load_progress_from_log",
    "load_samples",
]
