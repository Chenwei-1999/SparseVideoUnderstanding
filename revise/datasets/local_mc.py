"""Local multiple-choice dataset adapter for one-shot baselines.

This module owns the dataset-side behavior for local MC video-QA data such as
NExT-QA, EgoSchema-style JSON, and VideoEspresso. The CLI decides whether the
runtime is vLLM or another backend; the dataset adapter only knows how to format
questions, sample frames, and score answer letters.
"""

from __future__ import annotations

from typing import Any, Optional

from revise.pnp.protocols import LoopConfig
from revise.pnp.utils import (
    extract_frames,
    extract_video_info,
    format_question_block,
    format_videoespresso_question_block,
    normalize_answer_letter,
    sample_uniform_indices,
)

__all__ = ["LocalMCDataset", "build_oneshot_user_text"]


def build_oneshot_user_text(question_block: str, frame_indices: list[int]) -> str:
    lines = [question_block, ""]
    lines.append(f"You will be shown {len(frame_indices)} video frames.")
    lines.append("Answer with EXACTLY ONE option letter (for example: A/B/C/D/E). Do not output any other text.")
    lines.append("")
    lines.append("Frames:")
    for idx in frame_indices:
        lines.append(f"Frame {idx}: <image>")
    return "\n".join(lines)


class LocalMCDataset:
    """Single-round adapter for local multiple-choice video-QA datasets."""

    def __init__(
        self,
        *,
        dataset_name: str,
        videoespresso_use_official_prompt: bool,
        videoespresso_with_evidence: bool,
    ) -> None:
        self.dataset_name = str(dataset_name).strip().lower()
        self.videoespresso_use_official_prompt = bool(videoespresso_use_official_prompt)
        self.videoespresso_with_evidence = bool(videoespresso_with_evidence)

    def video_path(self, sample: Any) -> str:
        return sample.video_path

    def frame_count(self, sample: Any) -> int:
        frame_count, _ = extract_video_info(sample.video_path)
        return int(frame_count or 0)

    def num_choices(self, sample: Any) -> int:
        return len(sample.choices)

    def normalize_answer(self, sample: Any, answer_text: str) -> Optional[str]:
        return normalize_answer_letter(answer_text, self.num_choices(sample))

    def ground_truth_letter(self, sample: Any) -> Optional[str]:
        return normalize_answer_letter(chr(ord("A") + int(sample.answer_idx)), self.num_choices(sample))

    def is_correct(self, sample: Any, pred_letter: str) -> bool:
        gt = self.ground_truth_letter(sample)
        return bool(pred_letter and gt and pred_letter == gt)

    def format_question(self, sample: Any) -> str:
        if self.dataset_name == "videoespresso" and self.videoespresso_use_official_prompt:
            return format_videoespresso_question_block(
                sample.question,
                sample.choices,
                task=getattr(sample, "task", ""),
                evidence=getattr(sample, "evidence", ""),
                with_evidence=bool(self.videoespresso_with_evidence),
                revise_answer_tags=False,
            )
        return format_question_block(sample.question, sample.choices)

    def initial_frame_indices(self, sample: Any, frame_count: int, cfg: LoopConfig) -> list[int]:
        _ = sample
        return sample_uniform_indices(int(frame_count or 0), int(cfg.max_frames_per_round))

    def extract_frames(self, sample: Any, indices: list[int]) -> list[Any]:
        return extract_frames(sample.video_path, indices)

    def oneshot_user_text(
        self,
        question_block: str,
        num_frames: int,
        *,
        frame_indices: Optional[list[int]] = None,
    ) -> str:
        _ = num_frames
        return build_oneshot_user_text(question_block, list(frame_indices or []))
