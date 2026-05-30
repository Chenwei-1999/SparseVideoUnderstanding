#!/usr/bin/env python3
"""REVISE multi-round plug-and-play evaluation for Video-MME / LVBench (HF backend).

In-process HuggingFace ``transformers`` variant of the Video-MME / LVBench
pipeline: instead of talking to a vLLM server it loads the model and processor
directly (with a LLaVA-OneVision/Qwen routing path via
``revise/llava_next_runtime.py``) and runs the same REVISE multi-round loop
(``<think>`` -> ``<summarize>`` P/O/H/U/R -> ``<select>``/``<answer>``).

Use this backend for checkpoints that lack a vLLM serving path. Run as a CLI
(see ``main``); invoked by ``scripts/paper_suite.py``. Shared helpers live in
``revise/pnp_utils.py``.

NOTE: ``MCVideoSample`` / loaders here intentionally differ from the vLLM
variant in ``plug_and_play_videomme_lvbench_vllm.py`` (this one filters rows with
an empty answer and carries ``question_type``/``video_type`` rather than a
``video_url``). Deliberately not unified -- see
``tests/test_pnp_characterization.py::LoaderDivergenceTest``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

# Allow direct execution via `python examples/...py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise import pnp_engine
from revise.pnp_protocols import LoopConfig, RunStats
from revise.pnp_utils import FORCE_ANSWER_INSTRUCTIONS_SIMPLE as _FORCE_ANSWER_INSTRUCTIONS
from revise.pnp_utils import (
    ANSWER_RE,
    SELECT_RE,
    SUMMARIZE_RE,
    THINK_RE,
    apply_processor_chat_template,
    configure_llava_processor,
    dedupe_preserve_order,
    ensure_writable_hf_cache,
    extract_frames_1fps,
    extract_tag,
    extract_video_info,
    format_question_block,
    format_videomme_question_block,
    maybe_init_wandb,
    maybe_log_jsonl,
    normalize_answer_letter,
    parse_int_list,
    parse_options_from_lvbench_question,
    parse_time_reference_range,
    propose_candidate_frames,
    retry_feedback_text,
    sample_uniform_indices_inclusive,
    shard_by_video,
    stable_sample_id_dataset,
    timeline_len_1fps,
    wandb_log,
)
from revise.llava_next_runtime import is_llava_qwen_checkpoint, load_llava_next_runtime

ensure_writable_hf_cache(REPO_ROOT / "data" / "revise_assets" / "hf_home")
from datasets import load_dataset

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


_SYSTEM_PROMPT = (
    "You are a video question-answering agent. You may request more frames or answer now.\n"
    "EVERY response MUST begin with a <think>...</think> reasoning trace.\n"
    "On a Select round, follow <think> with <summarize> then <select>.\n"
    "On the Answer round, follow <think> with <answer> only (reuse your last committed summary).\n\n"
    "Example request:\n"
    "<think>...</think>\n"
    "<summarize>P: ...; O: ...; H: ...; U: ...; R: request additional frames</summarize>\n"
    "<select>1, 3</select>\n\n"
    "Example answer:\n"
    "<think>...</think>\n"
    "<answer>B</answer>\n\n"
    "Rules:\n"
    "- Every response begins with <think>...</think>.\n"
    "- <summarize>: the ONLY persistent memory across rounds. Keep it short and update it EVERY Select round.\n"
    "- In <select>, output comma-separated integers only (no brackets, no text).\n"
    "- In <answer>, output EXACTLY ONE option letter shown in the question (e.g., A/B/C/D/E). No words.\n"
    "- When Candidate Frame IDs are provided, output those IDs (1..K) in <select> instead of raw frame indices.\n"
)


def _prep_image(img: Image.Image, *, max_edge: int = 512) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        img = img.copy()
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return img


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
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def _idx_to_letters(idx: int) -> str:
        # Excel-style: 0->A, 25->Z, 26->AA, ...
        if idx < 0:
            return "?"
        base = len(letters)
        n = idx + 1
        out = ""
        while n > 0:
            n -= 1
            n, rem = divmod(n, base)
            out = letters[rem] + out
        return out

    allowed_letters = ", ".join(list(letters[: max(1, question_block.count("\n") - 1)]))  # heuristic

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
        lines.append("In <select>, output ONLY candidate IDs (comma-separated). Do NOT output raw indices when IDs exist.")
        if require_candidate_frames:
            lines.append("IMPORTANT: You MUST choose frames only from the Candidate IDs.")
    lines.extend(["Current summary:", f"<summarize>{summary}</summarize>", "Frames shown in this round:"])
    for i in range(len(current_frames)):
        lines.append(f"Shown frame {_idx_to_letters(i)} <image>")
    return "\n".join(lines)


def _build_chat_messages(system_prompt: str, user_text: str, images: list[Image.Image]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    content: list[dict[str, Any]] = []
    parts = user_text.split("<image>") if images else [user_text]
    if images and (len(parts) - 1) == len(images):
        for i, img in enumerate(images):
            if parts[i]:
                content.append({"type": "text", "text": parts[i]})
            content.append({"type": "image"})
        if parts[-1]:
            content.append({"type": "text", "text": parts[-1]})
    else:
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": user_text})

    conv: list[dict[str, Any]] = []
    if system_prompt:
        conv.append({"role": "system", "content": system_prompt})
    conv.append({"role": "user", "content": content})
    # Processor expects images list aligned with "image" placeholders.
    prepped_images = [_prep_image(img) for img in images]
    return conv, prepped_images


def _chat_once_hf(
    model: Any,
    processor: Any,
    system_prompt: str,
    user_text: str,
    images: list[Image.Image],
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_len: int,
) -> str:
    if hasattr(model, "chat_once"):
        return str(
            model.chat_once(
                system_prompt=system_prompt,
                user_text=user_text,
                images=[_prep_image(img) for img in images],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

    if processor is None:
        raise RuntimeError("hf_processor_missing: processor is required for transformers AutoModel generation")

    conv, prepped_images = _build_chat_messages(system_prompt, user_text, images)
    chat = apply_processor_chat_template(processor, conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, images=prepped_images, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[-1])
    if input_len + max_new_tokens > max_len:
        raise RuntimeError(f"prompt_too_long: input_len={input_len} max_len={max_len} max_new={max_new_tokens}")

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if temperature and temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
    else:
        gen_kwargs.update({"do_sample": False})
    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    gen_ids = generated[:, input_len:]
    return str(processor.batch_decode(gen_ids, skip_special_tokens=True)[0])


def _retry_feedback_text(feedback: str, *, force_answer: bool) -> str:
    return retry_feedback_text(
        feedback,
        force_answer=force_answer,
        force_instructions=_FORCE_ANSWER_INSTRUCTIONS,
    )


@dataclass
class MCVideoSample:
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


def _load_videomme_samples(split: str) -> list[MCVideoSample]:
    ds = load_dataset("lmms-lab/Video-MME", split=split)
    samples: list[MCVideoSample] = []
    for ex in ds:
        video_id = str(ex.get("videoID") or ex.get("video_id") or "").strip()
        qid = str(ex.get("question_id") or ex.get("qid") or "").strip()
        question = str(ex.get("question") or "").strip()
        options_raw = ex.get("options") or []
        if not isinstance(options_raw, list):
            options_raw = []
        options: list[str] = []
        for opt in options_raw:
            s = str(opt).strip()
            m = re.match(r"^[A-Z]\s*[.)]\s*(.*)$", s)
            options.append(m.group(1).strip() if m else s)
        answer = str(ex.get("answer") or "").strip().upper()
        if not video_id or not answer:
            continue
        samples.append(
            MCVideoSample(
                dataset="videomme",
                uid=qid or stable_sample_id_dataset("videomme", video_id, question),
                video_key=f"{video_id}.mp4",
                question=question,
                options=options,
                answer_letter=answer,
                time_reference="",
                question_type=str(ex.get("domain") or ex.get("sub_category") or "").strip(),
                video_type=str(ex.get("duration") or "").strip(),
            )
        )
    return samples


def _load_lvbench_samples(split: str) -> list[MCVideoSample]:
    ds = load_dataset("lmms-lab/LVBench", split=split)
    samples: list[MCVideoSample] = []
    for ex in ds:
        video_path = str(ex.get("video_path") or "").strip()
        uid = str(ex.get("uid") or ex.get("key") or "").strip() or video_path
        q_raw = str(ex.get("question") or "").strip()
        q_text, options = parse_options_from_lvbench_question(q_raw)
        answer = str(ex.get("answer") or "").strip().upper()
        time_reference = str(ex.get("time_reference") or "").strip()
        q_type = str(ex.get("question_type") or "").strip()
        v_type = str(ex.get("type") or "").strip()
        if not video_path or not answer:
            continue
        samples.append(
            MCVideoSample(
                dataset="lvbench",
                uid=uid,
                video_key=video_path,
                question=q_text if q_text else q_raw,
                options=options,
                answer_letter=answer,
                time_reference=time_reference,
                question_type=q_type,
                video_type=v_type,
            )
        )
    return samples


def _load_model_and_processor(model_path: str, dtype: str, device: torch.device) -> tuple[Any, Any]:
    torch_dtype = torch.bfloat16
    if dtype == "float16":
        torch_dtype = torch.float16
    if dtype == "float32":
        torch_dtype = torch.float32

    if is_llava_qwen_checkpoint(model_path):
        return load_llava_next_runtime(model_path, dtype, device), None

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    configure_llava_processor(processor, model_config)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model, processor


class LVBenchHFDataset:
    """HF-side Video-MME/LVBench adapter for the shared PnP engine.

    The sample loaders remain intentionally distinct from the vLLM long-video
    loader: HF filters rows without answers and keeps question/video type
    metadata instead of download URLs.
    """

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
        self._current_sample: Optional[MCVideoSample] = None
        self._frame_count_cache: dict[str, int] = {}

    def _cache_path(self, sample: MCVideoSample) -> Path:
        return Path(self.video_cache_dir) / sample.dataset / sample.video_key

    def video_path(self, sample: MCVideoSample) -> str:
        return str(self._cache_path(sample))

    def frame_count(self, sample: MCVideoSample) -> int:
        cached = self._frame_count_cache.get(sample.sample_id)
        if cached is not None:
            return cached
        total_frames, fps = extract_video_info(self.video_path(sample))
        timeline_len = timeline_len_1fps(total_frames, fps)
        if timeline_len <= 0:
            raise RuntimeError("invalid_video_timeline")
        self._frame_count_cache[sample.sample_id] = timeline_len
        return timeline_len

    def video_id(self, sample: MCVideoSample) -> str:
        return sample.video_key

    def num_choices(self, sample: MCVideoSample) -> int:
        return len(sample.options)

    def normalize_answer(self, sample: MCVideoSample, answer_text: str) -> Optional[str]:
        return normalize_answer_letter(answer_text, self.num_choices(sample))

    def ground_truth_letter(self, sample: MCVideoSample) -> Optional[str]:
        return normalize_answer_letter(sample.answer_letter, self.num_choices(sample))

    def is_correct(self, sample: MCVideoSample, pred_letter: str) -> bool:
        return pred_letter == self.ground_truth_letter(sample)

    def log_fields(self, sample: MCVideoSample) -> dict[str, Any]:
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

    def format_question(self, sample: MCVideoSample) -> str:
        self._current_sample = sample
        if sample.dataset == "videomme" and self.videomme_use_official_prompt:
            return format_videomme_question_block(sample.question, sample.options)
        return format_question_block(sample.question, sample.options)

    def system_prompt(self, cfg: LoopConfig) -> str:
        _ = cfg
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

    def extract_frames(self, sample: MCVideoSample, indices: list[int]) -> list[Image.Image]:
        return extract_frames_1fps(self.video_path(sample), indices)

    def sample_unseen_frames(self, frame_count: int, seen: set[int], k: int, rng: random.Random) -> list[int]:
        if frame_count <= 0 or k <= 0:
            return []
        candidates = [i for i in range(frame_count) if i not in seen]
        if not candidates:
            return []
        return sorted(rng.sample(candidates, k=min(k, len(candidates))))

    def _active_range(self, sample: MCVideoSample, frame_count: int) -> tuple[int, int]:
        if sample.time_reference:
            parsed = parse_time_reference_range(sample.time_reference, frame_count)
            if parsed is not None:
                return parsed
        return 0, max(0, int(frame_count) - 1)

    def initial_frame_indices(self, sample: MCVideoSample, frame_count: int, cfg: LoopConfig) -> list[int]:
        start, end = self._active_range(sample, frame_count)
        return sample_uniform_indices_inclusive(start, end, int(cfg.max_frames_per_round))

    def candidate_frame_indices(
        self,
        sample: MCVideoSample,
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

    def fallback_frame_indices(self, sample: MCVideoSample, frame_count: int, k: int, cfg: LoopConfig) -> list[int]:
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
            "This is the final round. You MUST answer now using <think>...</think> then "
            "<think>...</think> then <answer>LETTER</answer>."
        )

    def final_answer_instruction(self, cfg: LoopConfig) -> Optional[str]:
        _ = cfg
        return "You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."

    def forced_answer_request(
        self,
        sample: MCVideoSample,
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


class HFInProcessBackend:
    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        device: torch.device,
        max_len: int,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.max_len = int(max_len)

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
    ) -> str:
        _ = base_url, model_id, timeout_s
        return _chat_once_hf(
            model=self.model,
            processor=self.processor,
            system_prompt=system_prompt,
            user_text=user_text,
            images=images,
            device=self.device,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            max_len=self.max_len,
        )

    def get_model_id(self, base_url: str, model_id: Optional[str] = None) -> str:
        _ = base_url
        return model_id or "hf-in-process"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["videomme", "lvbench"], default="lvbench")
    ap.add_argument("--split", default="")
    ap.add_argument("--video-cache-dir", default="./data/revise_assets/video_cache",
                    help="Local cache for downloaded benchmark videos (set REVISE_VIDEO_CACHE_DIR or pass to override)")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=0)

    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--system-prompt", default=_SYSTEM_PROMPT)

    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-frames-per-round", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--use-candidate-frame-ids", action="store_true", default=True)
    ap.add_argument("--require-candidate-frames", action="store_true", default=True)
    ap.add_argument("--max-retries-per-round", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--force-final-answer", action="store_true", default=True)
    ap.add_argument(
        "--videomme-use-official-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the official no-subtitle Video-MME multiple-choice prompt template.",
    )

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)

    ap.add_argument("--log-jsonl", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--resume-from-log", action="store_true")

    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "revise_benchmarks"))
    ap.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    ap.add_argument("--wandb-name", default=os.getenv("WANDB_RUN_NAME"))
    ap.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", ""))
    ap.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", ""))

    args = ap.parse_args()

    split = args.split or ("test" if args.dataset == "videomme" else "train")
    if args.dataset == "videomme":
        samples = _load_videomme_samples(split)
    else:
        samples = _load_lvbench_samples(split)
    start_idx = max(0, int(args.start_idx or 0))
    end_idx = int(args.end_idx or 0)
    if end_idx <= 0:
        end_idx = len(samples)
    samples = samples[start_idx:end_idx]
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    samples = shard_by_video(samples, args.num_shards, args.shard_idx)
    samples.sort(key=lambda s: (s.video_key, s.uid))
    if not samples:
        raise SystemExit("No samples selected (check --split/--start-idx/--max-samples/--sharding).")

    if args.num_shards > 1:
        suffix = f".shard{args.shard_idx}of{args.num_shards}"

        def _suffix_path(path: str) -> str:
            root, ext = os.path.splitext(path)
            return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"

        if args.log_jsonl and suffix not in args.log_jsonl:
            args.log_jsonl = _suffix_path(args.log_jsonl)
        if args.summary_json and suffix not in args.summary_json:
            args.summary_json = _suffix_path(args.summary_json)

    resume_completed = 0
    if args.resume_from_log and args.log_jsonl and os.path.exists(args.log_jsonl):
        seen_samples: set[str] = set()
        with open(args.log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("sample_id")
                if sid and obj.get("answer_letter"):
                    seen_samples.add(str(sid))
        resume_completed = len(seen_samples)
        if resume_completed > 0:
            print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}")
    if resume_completed > 0:
        samples = samples[resume_completed:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = _load_model_and_processor(args.model_path, args.dtype, device)
    max_len = int(getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", 32768))

    run_config = {
        "task": "revise_plug_and_play_lvbench_hf",
        "dataset": args.dataset,
        "split": split,
        "model_path": args.model_path,
        "max_rounds": args.max_rounds,
        "max_frames_per_round": args.max_frames_per_round,
        "candidate_k": args.candidate_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    run = maybe_init_wandb(args, run_config)

    rng = random.Random(1337 + int(args.shard_idx))
    stats = RunStats()
    dataset_adapter = LVBenchHFDataset(
        split=split,
        video_cache_dir=args.video_cache_dir,
        system_prompt=str(args.system_prompt or ""),
        videomme_use_official_prompt=bool(args.videomme_use_official_prompt),
    )
    backend = HFInProcessBackend(model=model, processor=processor, device=device, max_len=max_len)
    cfg = LoopConfig(
        max_rounds=args.max_rounds,
        max_frames_per_round=args.max_frames_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        request_timeout_s=0,
        max_retries_per_round=args.max_retries_per_round,
        strict_actions=False,
        force_final_answer=bool(args.force_final_answer),
        use_candidate_frames=True,
        candidate_k=int(args.candidate_k),
        use_candidate_frame_ids=bool(args.use_candidate_frame_ids),
        require_candidate_frames=bool(args.require_candidate_frames),
        answer_only_final_round=False,
        observation_mode="image",
        caption_include="none",
        caption_max_chars=0,
        captions_dir=None,
        hide_seen_frames_in_prompt=False,
        log_jsonl=args.log_jsonl or None,
        seed=1337 + int(args.shard_idx),
    )

    correct = 0
    invalid_action_terminated = 0
    total_rounds = 0
    total_effective_rounds = 0
    total_frames_used = 0

    cache_dir = Path(args.video_cache_dir) / args.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for sample in samples:
        processed += 1
        processed_global = processed + resume_completed
        video_path = dataset_adapter.video_path(sample)

        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            stats.failed += 1
            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "error": "missing_video",
                },
            )
            continue

        try:
            outcome = pnp_engine.run_sample(
                sample,
                dataset=dataset_adapter,
                backend=backend,
                cfg=cfg,
                stats=stats,
                rng=rng,
                base_url="",
                model_id=backend.get_model_id("", model_id=args.model_path),
                run=run,
            )
        except Exception as e:
            stats.failed += 1
            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": split,
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "error": f"{type(e).__name__}: {str(e)[:400]}",
                },
            )
            continue

        if outcome.answer_letter is not None:
            total_rounds += min(outcome.round_idx, int(args.max_rounds))
            total_effective_rounds += outcome.effective_rounds
            total_frames_used += int(outcome.answer_frame_count)
            if dataset_adapter.is_correct(sample, outcome.answer_letter):
                correct += 1
        elif args.force_final_answer:
            invalid_action_terminated += 1

        failed = stats.failed
        invalid_outputs = stats.invalid_outputs
        total_model_calls = stats.total_model_calls
        total_retries = stats.total_retries
        think_present = dataset_adapter.think_present

        if run is not None:
            wandb_log(
                run,
                {
                    "progress/processed": processed_global,
                    "metrics/acc_so_far": correct / max(1, processed_global - failed),
                    "metrics/failed": failed,
                    "metrics/invalid_outputs": invalid_outputs,
                    "metrics/avg_rounds": total_rounds / max(1, processed_global - failed),
                    "metrics/avg_effective_rounds": total_effective_rounds / max(1, processed_global - failed),
                    "metrics/avg_frames_used": total_frames_used / max(1, processed_global - failed),
                },
                step=processed_global,
            )

        if processed_global % 25 == 0 or processed_global == len(samples):
            acc_so_far = correct / max(1, processed_global - failed)
            print(
                f"[{args.dataset}] processed {processed_global} / {len(samples)+resume_completed} | "
                f"acc={acc_so_far:.4f} failed={failed} invalid={invalid_outputs} calls={total_model_calls}"
            )

    failed = stats.failed
    invalid_outputs = stats.invalid_outputs
    total_model_calls = stats.total_model_calls
    total_retries = stats.total_retries
    think_present = dataset_adapter.think_present

    answered = max(0, len(samples) + resume_completed - failed)
    summary = {
        "dataset": args.dataset,
        "split": split,
        "model_path": args.model_path,
        "num_samples": len(samples) + resume_completed,
        "answered": answered,
        "correct": correct,
        "accuracy": correct / max(1, answered),
        "failed": failed,
        "invalid_outputs": invalid_outputs,
        "invalid_action_terminated": invalid_action_terminated,
        "avg_rounds": total_rounds / max(1, answered),
        "avg_effective_rounds": total_effective_rounds / max(1, answered),
        "avg_frames_used": total_frames_used / max(1, answered),
        "think_present": think_present,
        "total_model_calls": total_model_calls,
        "total_retries": total_retries,
        "max_len": max_len,
        "config": run_config,
        "ts": time.time(),
    }

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
