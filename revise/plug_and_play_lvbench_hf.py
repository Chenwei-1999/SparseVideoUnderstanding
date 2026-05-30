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
        force_instructions="You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>.",
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

    correct = 0
    failed = 0
    invalid_outputs = 0
    invalid_action_terminated = 0
    total_rounds = 0
    total_effective_rounds = 0
    total_frames_used = 0
    total_model_calls = 0
    total_retries = 0
    think_present = 0

    cache_dir = Path(args.video_cache_dir) / args.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for sample in samples:
        processed += 1
        processed_global = processed + resume_completed

        video_path = str(cache_dir / sample.video_key)
        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            failed += 1
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
            total_frames, fps = extract_video_info(video_path)
            timeline_len = timeline_len_1fps(total_frames, fps)
        except Exception as e:
            failed += 1
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
                    "error": f"video_probe_failed: {type(e).__name__}: {str(e)[:400]}",
                },
            )
            continue

        if timeline_len <= 0:
            failed += 1
            continue

        time_range = None
        if sample.time_reference:
            time_range = parse_time_reference_range(sample.time_reference, timeline_len)
        if time_range is None:
            range_start, range_end = 0, timeline_len - 1
        else:
            range_start, range_end = time_range

        if sample.dataset == "videomme" and args.videomme_use_official_prompt:
            question_block = format_videomme_question_block(sample.question, sample.options)
        else:
            question_block = format_question_block(sample.question, sample.options)
        summary_state = (
            "P: the agent has not seen any frames yet; "
            "O: no reliable observation yet; "
            "H: my belief will be updated based on what is observed; "
            "U: key detail is still unclear; "
            "R: need evidence from frames"
        )
        seen_frames: list[int] = []
        answer_letter: Optional[str] = None
        effective_rounds = 0
        terminated_invalid = False

        init_frames = sample_uniform_indices_inclusive(range_start, range_end, int(args.max_frames_per_round))
        next_frames = [int(i) for i in init_frames if i >= 0]

        for round_idx in range(1, int(args.max_rounds) + 1):
            frames_this_round = [i for i in next_frames if i not in seen_frames]
            if not frames_this_round:
                frames_this_round = sample_uniform_indices_inclusive(range_start, range_end, 1)
            frames_this_round = frames_this_round[: int(args.max_frames_per_round)]
            for i in frames_this_round:
                if i not in seen_frames:
                    seen_frames.append(i)

            local_len = max(0, range_end - range_start + 1)
            if local_len <= 0:
                candidate_next_frames = []
            else:
                seen_local = {int(i - range_start) for i in seen_frames if range_start <= i <= range_end}
                cand_local = propose_candidate_frames(
                    frame_count=local_len,
                    seen=seen_local,
                    k=int(args.candidate_k),
                    rng=rng,
                )
                candidate_next_frames = [int(i + range_start) for i in cand_local]

            images = extract_frames_1fps(video_path, frames_this_round)
            user_text = _build_user_text(
                question_block=question_block,
                summary=summary_state,
                timeline_len=timeline_len,
                round_idx=round_idx,
                current_frames=frames_this_round,
                seen_frames=seen_frames,
                candidate_unseen_frames=candidate_next_frames,
                use_candidate_frame_ids=bool(args.use_candidate_frame_ids),
                require_candidate_frames=bool(args.require_candidate_frames),
                time_reference=sample.time_reference,
            )
            if args.force_final_answer and round_idx >= int(args.max_rounds):
                user_text = (
                    f"{user_text}\n\n"
                    "This is the final round. You MUST answer now using <think>...</think> then "
                    "<think>...</think> then <answer>LETTER</answer>."
                )

            retry_feedback: Optional[str] = None
            raw_output = ""
            for retry_idx in range(int(args.max_retries_per_round) + 1):
                try:
                    raw_output = _chat_once_hf(
                        model=model,
                        processor=processor,
                        system_prompt=str(args.system_prompt or ""),
                        user_text=user_text
                        if retry_feedback is None
                        else _retry_feedback_text(
                            retry_feedback,
                            force_answer=bool(args.force_final_answer and round_idx >= int(args.max_rounds)),
                        ),
                        images=images,
                        device=device,
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_len=max_len,
                    )
                    total_model_calls += 1
                except Exception as e:
                    err_txt = f"{type(e).__name__}: {str(e)[:400]}"
                    retry_feedback = f"Model error; please retry. ({err_txt})"
                    total_retries += 1
                    if retry_idx < int(args.max_retries_per_round):
                        continue
                    failed += 1
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
                            "round_idx": round_idx,
                            "error": f"model_error: {err_txt}",
                        },
                    )
                    break

                # Paper protocol: every response MUST begin with a <think> trace.
                think = extract_tag(raw_output, THINK_RE)
                if think is None:
                    retry_feedback = (
                        "Invalid response: every response MUST begin with a <think>...</think> reasoning trace, "
                        "then either <summarize> + <select> (request) or <answer> (final)."
                    )
                    invalid_outputs += 1
                    total_retries += 1
                    continue
                think_present += 1

                summary = extract_tag(raw_output, SUMMARIZE_RE)
                frames_text = extract_tag(raw_output, SELECT_RE)
                answer = extract_tag(raw_output, ANSWER_RE)

                if answer is not None:
                    # Strict-paper Answer round: <think> + <answer> only; the last committed
                    # <summarize> (set on the prior Select round) is reused as the state.
                    letter = normalize_answer_letter(answer, len(sample.options))
                    if letter is None:
                        retry_feedback = "Invalid response: <answer> must be a single option letter."
                        invalid_outputs += 1
                        total_retries += 1
                        continue
                    answer_letter = letter
                    if summary is not None:
                        summary_state = summary
                    break

                # Select round: require a <summarize> then a <select> request.
                if summary is None:
                    retry_feedback = "Invalid response: missing <summarize> tag."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if frames_text is None:
                    retry_feedback = "Invalid response: missing <select> tag for requesting more frames."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                requested = dedupe_preserve_order(parse_int_list(frames_text))
                if not requested:
                    retry_feedback = "Invalid response: <select> must contain at least one integer."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if len(requested) > int(args.max_frames_per_round):
                    requested = requested[: int(args.max_frames_per_round)]
                    invalid_outputs += 1

                mapped = requested
                if args.use_candidate_frame_ids and candidate_next_frames:
                    mapped2: list[int] = []
                    allowed = set(int(x) for x in candidate_next_frames)
                    for cid in requested:
                        if 1 <= cid <= len(candidate_next_frames):
                            mapped2.append(int(candidate_next_frames[cid - 1]))
                        elif cid in allowed:
                            mapped2.append(int(cid))
                    mapped = mapped2
                if args.require_candidate_frames and candidate_next_frames:
                    allowed = set(int(x) for x in candidate_next_frames)
                    if any(x not in allowed for x in mapped):
                        retry_feedback = "Invalid response: requested frames must be within candidate IDs."
                        invalid_outputs += 1
                        total_retries += 1
                        continue

                if any(x in seen_frames for x in mapped):
                    retry_feedback = "Invalid response: requested frames must be NEW (unseen)."
                    invalid_outputs += 1
                    total_retries += 1
                    continue

                if mapped:
                    next_frames = mapped
                    effective_rounds += 1
                else:
                    next_frames = candidate_next_frames[: int(args.max_frames_per_round)]
                    invalid_outputs += 1

                summary_state = summary
                break

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
                    "timeline_len": timeline_len,
                    "round_idx": round_idx,
                    "retry_feedback": retry_feedback,
                    "question": sample.question,
                    "options": sample.options,
                    "answer_gt": sample.answer_letter,
                    "time_reference": sample.time_reference,
                    "question_type": sample.question_type,
                    "video_type": sample.video_type,
                    "seen_frames": seen_frames,
                    "current_frames": frames_this_round,
                    "candidate_unseen_frames": candidate_next_frames,
                    "summary_in": summary_state,
                    "raw_output": raw_output,
                    "answer_letter": answer_letter,
                },
            )

            if answer_letter is not None:
                total_rounds += round_idx
                total_effective_rounds += effective_rounds
                total_frames_used += len(frames_this_round)
                gt = normalize_answer_letter(sample.answer_letter, len(sample.options))
                if gt is not None and answer_letter == gt:
                    correct += 1
                break

        if answer_letter is None and args.force_final_answer:
            # If we still have no answer, force one extra call (answer-only).
            images = extract_frames_1fps(video_path, frames_this_round)
            user_text2 = (
                f"{question_block}\n"
                "You MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."
            )
            try:
                raw = _chat_once_hf(
                    model=model,
                    processor=processor,
                    system_prompt=str(args.system_prompt or ""),
                    user_text=user_text2,
                    images=images,
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_len=max_len,
                )
            except Exception as e:
                failed += 1
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
                        "round_idx": int(args.max_rounds),
                        "error": f"model_error_final_answer: {type(e).__name__}: {str(e)[:400]}",
                    },
                )
                continue
            total_model_calls += 1
            ans = extract_tag(raw, ANSWER_RE)
            summary = extract_tag(raw, SUMMARIZE_RE)
            letter = normalize_answer_letter(ans or "", len(sample.options))
            if letter is not None:
                answer_letter = letter
                total_rounds += int(args.max_rounds)
                total_effective_rounds += effective_rounds
                total_frames_used += len(frames_this_round)
                gt = normalize_answer_letter(sample.answer_letter, len(sample.options))
                if gt is not None and answer_letter == gt:
                    correct += 1
            else:
                terminated_invalid = True

        if terminated_invalid:
            invalid_action_terminated += 1

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
