"""Protocols and shared counters for plug-and-play evaluation engines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Protocol


class Dataset(Protocol):
    """Dataset-specific operations used by the shared plug-and-play loop."""

    def format_question(self, sample: Any) -> str: ...

    def system_prompt(self, cfg: "LoopConfig") -> str: ...

    def build_user_text(
        self,
        *,
        question_block: str,
        summary: str,
        frame_count: int,
        round_idx: int,
        frame_indices: list[int],
        seen_frames: list[int],
        render_images: bool = True,
        hide_seen_frames: bool = False,
        candidate_unseen_frames: Optional[list[int]] = None,
        use_candidate_frame_ids: bool = False,
        require_candidate_frames: bool = False,
        shown_frame_captions: Optional[list[str]] = None,
        candidate_id_captions: Optional[list[str]] = None,
        shown_frame_ts: Optional[list[int]] = None,
        candidate_id_ts: Optional[list[int]] = None,
    ) -> str: ...

    def extract_frames(self, sample: Any, indices: list[int]) -> list[Any]: ...

    def sample_unseen_frames(
        self,
        frame_count: int,
        seen: set[int],
        k: int,
        rng: random.Random,
    ) -> list[int]: ...

    def retry_feedback_text(self, feedback: str, *, force_answer: bool = False) -> str: ...

    def load_video_captions(self, captions_dir: str, video_id: str) -> dict[int, str]: ...

    def get_video_fps(self, video_path: str) -> float: ...

    def caption_key_for_frame_index(self, frame_idx: int, fps: float) -> int: ...


class Backend(Protocol):
    """Model backend operation used by the shared plug-and-play loop."""

    def chat(
        self,
        *,
        base_url: str,
        model_id: str,
        system_prompt: str,
        user_text: str,
        images: list[Any],
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_s: int,
    ) -> str: ...

    def get_model_id(self, base_url: str, model_id: Optional[str] = None) -> str: ...


@dataclass
class RunStats:
    processed: int = 0
    correct: int = 0
    total_rounds: int = 0
    total_frames_used: int = 0
    effective_rounds_total: int = 0
    failed: int = 0
    invalid_outputs: int = 0
    invalid_action_terminated: int = 0
    total_retries: int = 0
    total_model_calls: int = 0
    fallback_frames_used: int = 0


@dataclass
class LoopConfig:
    max_rounds: int
    max_frames_per_round: int
    temperature: float
    top_p: float
    max_tokens: int
    request_timeout_s: int
    max_retries_per_round: int
    strict_actions: bool
    force_final_answer: bool
    use_candidate_frames: bool
    candidate_k: Optional[int]
    use_candidate_frame_ids: bool
    require_candidate_frames: bool
    answer_only_final_round: bool
    observation_mode: str
    caption_include: str
    caption_max_chars: int
    captions_dir: Optional[str]
    hide_seen_frames_in_prompt: bool
    log_jsonl: Optional[str]
    seed: int
