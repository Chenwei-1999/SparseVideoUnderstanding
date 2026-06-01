"""LVBench dataset loading and long-video task adapters."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from revise.benchmarks.videomme_lvbench_vllm import LVBenchDataset, MCVideoSample
from revise.benchmarks.videomme_lvbench_vllm import _load_lvbench_samples as load_samples
from revise.pnp.prompts import SYSTEM_PROMPT
from revise.pnp.protocols import LoopConfig
from revise.pnp.utils import (
    ANSWER_RE,
    OPTION_LABELS,
    SELECT_RE,
    SUMMARIZE_RE,
    THINK_RE,
    dedupe_preserve_order,
    ensure_writable_hf_cache,
    extract_frames_1fps,
    extract_tag,
    extract_video_info,
    format_question_block,
    format_videomme_question_block,
    normalize_answer_letter,
    parse_int_list,
    parse_options_from_lvbench_question,
    parse_time_reference_range,
    propose_candidate_frames,
    retry_feedback_text,
    sample_uniform_indices_inclusive,
    stable_sample_id_dataset,
    timeline_len_1fps,
)
from revise.pnp.utils import FORCE_ANSWER_INSTRUCTIONS_SIMPLE as _FORCE_ANSWER_INSTRUCTIONS

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CACHED_LONG_VIDEO_SYSTEM_PROMPT = SYSTEM_PROMPT

__all__ = [
    "CachedLongVideoDataset",
    "CachedMCVideoSample",
    "LVBenchDataset",
    "LVBenchHFDataset",
    "MCVideoSample",
    "DEFAULT_CACHED_LONG_VIDEO_SYSTEM_PROMPT",
    "load_dataset",
    "load_hf_samples",
    "load_samples",
    "load_videomme_hf_samples",
]


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Lazily import HF datasets after redirecting caches into the repo data root."""
    ensure_writable_hf_cache(REPO_ROOT / "data" / "revise_assets" / "hf_home")
    from datasets import load_dataset as _hf_load_dataset

    return _hf_load_dataset(*args, **kwargs)


@dataclass
class CachedMCVideoSample:
    """Long-video MC sample for cache-only/in-process runs."""

    dataset: str
    uid: str
    video_key: str
    question: str
    options: list[str]
    answer_letter: str
    time_reference: str
    question_type: str
    video_type: str

    @property
    def sample_id(self) -> str:
        return stable_sample_id_dataset(self.dataset, self.video_key, self.uid)


def _clean_options(options_raw: Any) -> list[str]:
    if not isinstance(options_raw, list):
        return []
    options: list[str] = []
    for opt in options_raw:
        s = str(opt).strip()
        m = re.match(r"^[A-Z]\s*[.)]\s*(.*)$", s)
        options.append(m.group(1).strip() if m else s)
    return options


def load_videomme_hf_samples(split: str) -> list[CachedMCVideoSample]:
    ds = load_dataset("lmms-lab/Video-MME", split=split)
    samples: list[CachedMCVideoSample] = []
    for ex in ds:
        video_id = str(ex.get("videoID") or ex.get("video_id") or "").strip()
        qid = str(ex.get("question_id") or ex.get("qid") or "").strip()
        question = str(ex.get("question") or "").strip()
        answer = str(ex.get("answer") or "").strip().upper()
        if not video_id or not answer:
            continue
        samples.append(
            CachedMCVideoSample(
                dataset="videomme",
                uid=qid or stable_sample_id_dataset("videomme", video_id, question),
                video_key=f"{video_id}.mp4",
                question=question,
                options=_clean_options(ex.get("options") or []),
                answer_letter=answer,
                time_reference="",
                question_type=str(ex.get("domain") or ex.get("sub_category") or "").strip(),
                video_type=str(ex.get("duration") or "").strip(),
            )
        )
    return samples


def load_hf_samples(split: str) -> list[CachedMCVideoSample]:
    ds = load_dataset("lmms-lab/LVBench", split=split)
    samples: list[CachedMCVideoSample] = []
    for ex in ds:
        video_path = str(ex.get("video_path") or "").strip()
        uid = str(ex.get("uid") or ex.get("key") or "").strip() or video_path
        q_raw = str(ex.get("question") or "").strip()
        q_text, options = parse_options_from_lvbench_question(q_raw)
        answer = str(ex.get("answer") or "").strip().upper()
        if not video_path or not answer:
            continue
        samples.append(
            CachedMCVideoSample(
                dataset="lvbench",
                uid=uid,
                video_key=video_path,
                question=q_text if q_text else q_raw,
                options=options,
                answer_letter=answer,
                time_reference=str(ex.get("time_reference") or "").strip(),
                question_type=str(ex.get("question_type") or "").strip(),
                video_type=str(ex.get("type") or "").strip(),
            )
        )
    return samples


def _build_user_text(
    question_block: str,
    summary: str,
    timeline_len: int,
    round_idx: int,
    current_frames: list[int],
    seen_frames: list[int],
    candidate_unseen_frames: list[int],
    use_candidate_frame_ids: bool,
    require_candidate_frames: bool,
    time_reference: str = "",
) -> str:
    def _idx_to_letters(idx: int) -> str:
        if idx < 0:
            return "?"
        base = len(OPTION_LABELS)
        n = idx + 1
        out = ""
        while n > 0:
            n -= 1
            n, rem = divmod(n, base)
            out = OPTION_LABELS[rem] + out
        return out

    allowed_letters = ", ".join(list(OPTION_LABELS[: max(1, question_block.count("\n") - 1)]))

    lines: list[str] = []
    lines.append(f"Round {round_idx} / Question:")
    lines.append(question_block)
    if allowed_letters:
        lines.append(
            f"To answer, output <think>...</think> then <answer>LETTER</answer> "
            f"(LETTER must be one of: {allowed_letters})."
        )
    lines.append(f"Total frames L = {timeline_len} (1 fps timeline).")
    if time_reference:
        lines.append(f"Relevant time window for this question: {time_reference} (focus on this segment).")
    lines.append(
        f"Seen frames: {len(seen_frames)} frames already viewed (do NOT request any previously shown frames)."
    )
    if use_candidate_frame_ids and candidate_unseen_frames:
        lines.append(
            f"Candidate unseen frames available as IDs (all NEW): choose IDs in [1, {len(candidate_unseen_frames)}]."
        )
        id_map = ", ".join(f"{i+1}->{t}s" for i, t in enumerate(candidate_unseen_frames))
        lines.append(f"Candidate ID -> timeline second: {id_map}")
        lines.append(
            "In <select>, output ONLY candidate IDs (comma-separated). "
            "Do NOT output raw indices when IDs exist."
        )
        if require_candidate_frames:
            lines.append("IMPORTANT: You MUST choose frames only from the Candidate IDs.")
    lines.extend(["Current summary:", f"<summarize>{summary}</summarize>", "Frames shown in this round:"])
    for i in range(len(current_frames)):
        lines.append(f"Shown frame {_idx_to_letters(i)} <image>")
    return "\n".join(lines)


def _retry_feedback_text(feedback: str, *, force_answer: bool) -> str:
    return retry_feedback_text(
        feedback,
        force_answer=force_answer,
        force_instructions=_FORCE_ANSWER_INSTRUCTIONS,
    )


class CachedLongVideoDataset:
    """Cache-only long-video adapter for in-process model backends."""

    def __init__(
        self,
        *,
        split: str,
        video_cache_dir: str,
        system_prompt: str,
        videomme_use_official_prompt: bool = True,
    ) -> None:
        self.split = split
        self.video_cache_dir = video_cache_dir
        self.system_prompt_text = system_prompt
        self.videomme_use_official_prompt = bool(videomme_use_official_prompt)
        self.think_present = 0
        self._current_sample: Optional[CachedMCVideoSample] = None
        self._frame_count_cache: dict[str, int] = {}

    def _cache_path(self, sample: CachedMCVideoSample) -> Path:
        return Path(self.video_cache_dir) / sample.dataset / sample.video_key

    def video_path(self, sample: CachedMCVideoSample) -> str:
        return str(self._cache_path(sample))

    def frame_count(self, sample: CachedMCVideoSample) -> int:
        cached = self._frame_count_cache.get(sample.sample_id)
        if cached is not None:
            return cached
        total_frames, fps = extract_video_info(self.video_path(sample))
        timeline_len = timeline_len_1fps(total_frames, fps)
        if timeline_len <= 0:
            raise RuntimeError("invalid_video_timeline")
        self._frame_count_cache[sample.sample_id] = timeline_len
        return timeline_len

    def video_id(self, sample: CachedMCVideoSample) -> str:
        return sample.video_key

    def num_choices(self, sample: CachedMCVideoSample) -> int:
        return len(sample.options)

    def normalize_answer(self, sample: CachedMCVideoSample, answer_text: str) -> Optional[str]:
        return normalize_answer_letter(answer_text, self.num_choices(sample))

    def ground_truth_letter(self, sample: CachedMCVideoSample) -> Optional[str]:
        return normalize_answer_letter(sample.answer_letter, self.num_choices(sample))

    def is_correct(self, sample: CachedMCVideoSample, pred_letter: str) -> bool:
        return pred_letter == self.ground_truth_letter(sample)

    def log_fields(self, sample: CachedMCVideoSample) -> dict[str, Any]:
        return {
            "dataset": sample.dataset,
            "split": self.split,
            "sample_id": sample.sample_id,
            "uid": sample.uid,
            "video_key": sample.video_key,
            "video_id": sample.video_key,
            "video_path": self.video_path(sample),
            "question": sample.question,
            "options": sample.options,
            "answer_gt": sample.answer_letter,
            "time_reference": sample.time_reference,
            "question_type": sample.question_type,
            "video_type": sample.video_type,
        }

    def format_question(self, sample: CachedMCVideoSample) -> str:
        self._current_sample = sample
        if sample.dataset == "videomme" and self.videomme_use_official_prompt:
            return format_videomme_question_block(sample.question, sample.options)
        return format_question_block(sample.question, sample.options)

    def system_prompt(self, cfg: LoopConfig) -> str:
        if "{max_frames_per_round}" in self.system_prompt_text:
            return self.system_prompt_text.format(max_frames_per_round=cfg.max_frames_per_round)
        return self.system_prompt_text

    def build_user_text(self, **kwargs: Any) -> str:
        sample = self._current_sample
        return _build_user_text(
            question_block=kwargs["question_block"],
            summary=kwargs["summary"],
            timeline_len=kwargs["frame_count"],
            round_idx=kwargs["round_idx"],
            current_frames=kwargs["frame_indices"],
            seen_frames=kwargs["seen_frames"],
            candidate_unseen_frames=kwargs.get("candidate_unseen_frames") or [],
            use_candidate_frame_ids=bool(kwargs.get("use_candidate_frame_ids", False)),
            require_candidate_frames=bool(kwargs.get("require_candidate_frames", False)),
            time_reference=sample.time_reference if sample is not None else "",
        )

    def extract_frames(self, sample: CachedMCVideoSample, indices: list[int]) -> list[Image.Image]:
        return extract_frames_1fps(self.video_path(sample), indices)

    def oneshot_user_text(
        self,
        question_block: str,
        num_frames: int,
        *,
        frame_indices: Optional[list[int]] = None,
    ) -> str:
        _ = num_frames, frame_indices
        return (
            f"{question_block}\n\n"
            "You will be given video frames sampled at 1 fps.\n"
            "Answer with EXACTLY ONE option letter (e.g., A/B/C/D). Do not output any other text."
        )

    def sample_unseen_frames(self, frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
        if frame_count <= 0 or k <= 0:
            return []
        candidates = [i for i in range(frame_count) if i not in seen]
        if not candidates:
            return []
        return sorted(rng.sample(candidates, k=min(k, len(candidates))))

    def _active_range(self, sample: CachedMCVideoSample, frame_count: int) -> tuple[int, int]:
        if sample.time_reference:
            parsed = parse_time_reference_range(sample.time_reference, frame_count)
            if parsed is not None:
                return parsed
        return 0, max(0, int(frame_count) - 1)

    def initial_frame_indices(self, sample: CachedMCVideoSample, frame_count: int, cfg: LoopConfig) -> list[int]:
        start, end = self._active_range(sample, frame_count)
        return sample_uniform_indices_inclusive(start, end, int(cfg.max_frames_per_round))

    def candidate_frame_indices(
        self,
        sample: CachedMCVideoSample,
        *,
        frame_count: int,
        seen_frames: list[int],
        k: int,
        rng: random.Random,
    ) -> list[int]:
        start, end = self._active_range(sample, frame_count)
        local_len = max(0, end - start + 1)
        if local_len <= 0:
            return []
        seen_local = {int(i - start) for i in seen_frames if start <= int(i) <= end}
        cand_local = propose_candidate_frames(frame_count=local_len, seen=seen_local, k=int(k), rng=rng)
        return [int(i + start) for i in cand_local]

    def fallback_frame_indices(
        self, sample: CachedMCVideoSample, frame_count: int, k: int, cfg: LoopConfig
    ) -> list[int]:
        _ = cfg
        start, end = self._active_range(sample, frame_count)
        return sample_uniform_indices_inclusive(start, end, int(k))

    def retry_feedback_text(
        self,
        reason: str,
        *,
        force_answer: bool = False,
        max_frames_per_round: int = 0,
        frame_count: int = 0,
        seen_frames: Optional[list[int]] = None,
    ) -> str:
        _ = max_frames_per_round, frame_count, seen_frames
        messages = {
            "missing_think": (
                "Invalid response: every response MUST begin with a <think>...</think> reasoning trace, "
                "then either <summarize> + <select> (request) or <answer> (final)."
            ),
            "invalid_answer_letter": "Invalid response: <answer> must be a single option letter.",
            "invalid_select_summary": "Invalid response: missing <summarize> tag.",
            "missing_frames_tag": "Invalid response: missing <select> tag for requesting more frames.",
            "invalid_frames": "Invalid response: <select> must contain at least one integer.",
            "frames_not_in_candidates": "Invalid response: requested frames must be within candidate IDs.",
            "frames_already_seen": "Invalid response: requested frames must be NEW (unseen).",
        }
        return _retry_feedback_text(messages.get(reason, reason), force_answer=force_answer)

    def load_video_captions(self, captions_dir: str, video_id: str) -> dict[int, str]:
        _ = captions_dir, video_id
        return {}

    def get_video_fps(self, video_path: str) -> float:
        _ = video_path
        return 0.0

    def caption_key_for_frame_index(self, frame_idx: int, fps: float) -> int:
        _ = fps
        return int(frame_idx)

    def parse_think(self, raw: str) -> Optional[str]:
        think = extract_tag(raw, THINK_RE)
        if think is not None:
            self.think_present += 1
        return think

    def parse_summary(self, raw: str) -> Optional[str]:
        return extract_tag(raw, SUMMARIZE_RE)

    def parse_answer(self, raw: str) -> Optional[str]:
        return extract_tag(raw, ANSWER_RE)

    def parse_select(self, raw: str) -> Optional[str]:
        return extract_tag(raw, SELECT_RE)

    def should_commit_summary(self, summary: Optional[str], *, seen_count: int) -> bool:
        _ = seen_count
        return summary is not None

    def is_select_summary_valid(self, summary: Optional[str], *, seen_count: int) -> bool:
        _ = seen_count
        return summary is not None

    def is_summary_stale(self, summary: Optional[str], *, seen_count: int) -> bool:
        _ = summary, seen_count
        return False

    def select_has_range_syntax(self, frames_text: str) -> bool:
        _ = frames_text
        return False

    def parse_select_frames(self, frames_text: str) -> list[int]:
        return dedupe_preserve_order(parse_int_list(frames_text))

    def map_candidate_frame_ids(
        self,
        requested_ids: list[int],
        candidate_frames: list[int],
    ) -> Optional[list[int]]:
        mapped: list[int] = []
        allowed = {int(x) for x in candidate_frames}
        for cid in requested_ids:
            if 1 <= int(cid) <= len(candidate_frames):
                mapped.append(int(candidate_frames[int(cid) - 1]))
            elif int(cid) in allowed:
                mapped.append(int(cid))
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
        _ = frame_count
        requested = [int(i) for i in requested_frames]
        if require_candidate_frames and candidate_frames:
            allowed = {int(i) for i in candidate_frames}
            if any(i not in allowed for i in requested):
                return [], "frames_not_in_candidates"
        if any(i in seen_frames for i in requested):
            return [], "frames_already_seen"
        return requested, None

    def initial_summary(self, cfg: LoopConfig) -> str:
        _ = cfg
        return (
            "P: the agent has not seen any frames yet; "
            "O: no reliable observation yet; "
            "H: my belief will be updated based on what is observed; "
            "U: key detail is still unclear; "
            "R: need evidence from frames"
        )

    def final_round_instruction(self, cfg: LoopConfig) -> Optional[str]:
        _ = cfg
        return (
            "This is the final round. You MUST answer now using <think>...</think> then <answer>LETTER</answer>."
        )

    def final_answer_instruction(self, cfg: LoopConfig) -> Optional[str]:
        _ = cfg
        return "You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."

    def forced_answer_request(
        self,
        sample: CachedMCVideoSample,
        *,
        question_block: str,
        frame_count: int,
        max_rounds: int,
        system_prompt: str,
        last_user_text: str,
        last_images: list[Any],
    ) -> tuple[str, str, list[Any]]:
        _ = sample, frame_count, max_rounds, system_prompt, last_user_text
        user_text = (
            f"{question_block}\n"
            "You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."
        )
        return self.system_prompt_text, user_text, last_images

    def should_terminate_on_invalid_summary(self, cfg: LoopConfig) -> bool:
        _ = cfg
        return False

    def should_fail_on_empty_images(self, cfg: LoopConfig) -> bool:
        _ = cfg
        return False

    def should_count_exhausted_invalid_as_retry(self, reason: str) -> bool:
        return reason in {
            "missing_think",
            "invalid_answer_letter",
            "invalid_select_summary",
            "missing_frames_tag",
            "frames_not_in_candidates",
            "frames_already_seen",
            "invalid_frames",
        }

    def should_clear_frame_plan_on_exhausted_invalid(self, reason: str) -> bool:
        return reason in {
            "missing_think",
            "invalid_answer_letter",
            "invalid_select_summary",
            "missing_frames_tag",
            "frames_not_in_candidates",
            "frames_already_seen",
            "invalid_frames",
        }

    def should_retry_invalid_output(self, reason: str) -> bool:
        return reason != "too_many_frames"


LVBenchHFDataset = CachedLongVideoDataset
