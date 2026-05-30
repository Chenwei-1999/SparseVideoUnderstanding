#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise import pnp_harness
from revise.pnp_protocols import LoopConfig, RunStats
from revise.pnp_utils import (
    OPTION_LABELS as _ANSWER_LETTERS,
    collapse_ws as _collapse_ws,
    configure_llava_processor,
    ensure_writable_hf_cache,
    maybe_init_wandb as _maybe_init_wandb,
    parse_options_from_lvbench_question as _parse_options_from_lvbench_question,
    shard_by_video as _shard_by_video,
    stable_sample_id_dataset as _stable_sample_id,
    wandb_log as _wandb_log,
)
from revise.plug_and_play_lvbench_hf import HFInProcessBackend, LVBenchHFDataset

ensure_writable_hf_cache(REPO_ROOT / "data" / "revise_assets" / "hf_home")

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor


def _normalize_answer_letter(ans: str, num_choices: int) -> Optional[str]:
    if not ans:
        return None
    a = _collapse_ws(ans).strip().upper()
    if len(a) == 1 and a in _ANSWER_LETTERS[: max(1, num_choices)]:
        return a
    m = re.search(r"\b([A-Z])\b", a)
    if m:
        cand = m.group(1)
        if cand in _ANSWER_LETTERS[: max(1, num_choices)]:
            return cand
    return None


def _ensure_yt_dlp(py_bin: str) -> list[str]:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [py_bin, "-m", "yt_dlp"]


def _download_youtube(url: str, out_mp4: str, *, py_bin: str, timeout_s: int) -> None:
    out_mp4_path = Path(out_mp4)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_mp4_path.with_suffix("")) + ".%(ext)s"

    node_path = shutil.which("node")
    js_runtime_args: list[str] = []
    if node_path:
        js_runtime_args = ["--js-runtimes", f"node:{node_path}"]

    cmd = [
        *_ensure_yt_dlp(py_bin),
        *js_runtime_args,
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "--extractor-args",
        "youtube:player_client=android",
        "-f",
        "best[ext=mp4][height<=480]/best[ext=mp4]/best",
        "-o",
        out_tmpl,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed ({proc.returncode}): {proc.stderr.strip()[:500]}")

    if out_mp4_path.exists() and out_mp4_path.stat().st_size > 0:
        return
    candidates = list(out_mp4_path.parent.glob(out_mp4_path.stem + ".*"))
    for c in candidates:
        if c.suffix.lower() == ".mp4" and c.stat().st_size > 0:
            c.rename(out_mp4_path)
            return
    raise FileNotFoundError(f"Downloaded file not found for {url} (expected {out_mp4_path})")


@dataclass
class MCVideoSample:
    dataset: str
    uid: str
    video_key: str
    video_url: str
    question: str
    options: list[str]
    answer_letter: str
    time_reference: str = ""

    @property
    def sample_id(self) -> str:
        return _stable_sample_id(self.dataset, self.video_key, self.uid)


def _load_lvbench_samples(split: str) -> list[MCVideoSample]:
    ds = load_dataset("lmms-lab/LVBench", split=split)
    samples: list[MCVideoSample] = []
    for ex in ds:
        video_path = str(ex.get("video_path") or "").strip()
        uid = str(ex.get("uid") or ex.get("key") or "").strip()
        q_raw = str(ex.get("question") or "").strip()
        q_text, options = _parse_options_from_lvbench_question(q_raw)
        answer = str(ex.get("answer") or "").strip().upper()
        time_reference = str(ex.get("time_reference") or "").strip()
        video_id = Path(video_path).stem
        url = f"https://www.youtube.com/watch?v={video_id}"
        samples.append(
            MCVideoSample(
                dataset="lvbench",
                uid=uid or _stable_sample_id("lvbench", video_path, q_raw),
                video_key=video_path,
                video_url=url,
                question=q_text if q_text else q_raw,
                options=options,
                answer_letter=answer,
                time_reference=time_reference,
            )
        )
    return samples


def _load_model_and_processor(model_path: str, dtype: str, device: torch.device) -> tuple[Any, Any]:
    torch_dtype: Any
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    configure_llava_processor(processor, model_config)
    model = None
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.eval()
    model.to(device)
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--video-cache-dir", default="./data/revise_assets/video_cache",
                    help="Local cache for downloaded benchmark videos (set REVISE_VIDEO_CACHE_DIR or pass to override)")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)

    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-frames", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--yt-dlp-timeout-s", type=int, default=600)

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)

    ap.add_argument("--log-jsonl", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--resume-from-log", action="store_true")

    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "verl-revise"))
    ap.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    ap.add_argument("--wandb-name", default=os.getenv("WANDB_RUN_NAME"))
    ap.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    ap.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", ""))
    ap.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", ""))

    args = ap.parse_args()

    split = args.split
    samples = _load_lvbench_samples(args.split)

    start_idx = max(0, int(args.start_idx or 0))
    end_idx = int(args.end_idx or 0)
    if end_idx <= 0:
        end_idx = len(samples)
    samples = samples[start_idx:end_idx]
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    samples = _shard_by_video(samples, args.num_shards, args.shard_idx)
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
                if sid and obj.get("pred_answer"):
                    seen_samples.add(sid)
        resume_completed = len(seen_samples)
        if resume_completed > 0:
            print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}")
    if resume_completed > 0:
        samples = samples[resume_completed:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = _load_model_and_processor(args.model_path, args.dtype, device)
    max_len = int(getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", 32768))

    run_config = {
        "task": "lvbench_oneshot_hf",
        "dataset": "lvbench",
        "split": args.split,
        "model_path": args.model_path,
        "video_cache_dir": args.video_cache_dir,
        "dtype": args.dtype,
        "max_frames": args.max_frames,
        "max_new_tokens": args.max_new_tokens,
        "max_len": max_len,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "max_samples": args.max_samples,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
    }
    run = _maybe_init_wandb(args, run_config)

    rng = random.Random(42 + int(args.shard_idx))

    # Pre-create the per-dataset cache dir so the existence-skip path matches the
    # legacy behavior (the legacy loop mkdir'd it per sample).
    cache_dir = Path(args.video_cache_dir) / "lvbench"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the shared multi-round HF adapter (LVBenchHFDataset) for the
    # single-round baseline: run_sample_oneshot only exercises
    # frame_count / format_question / initial_frame_indices / extract_frames /
    # oneshot_user_text / normalize_answer / is_correct, all of which it already
    # implements. The harness owns the video-existence skip (never downloads),
    # scoring, logging, and this launcher's own summary schema.
    #
    # Behavioral deltas vs. the legacy standalone loop (consistent with the
    # multi-round HF migration): no yt-dlp download / `.failed` markers, no 3x
    # inference retry, and no prompt-length truncation. Samples whose video is
    # absent are counted as failed (existence skip); model errors propagate.
    dataset_adapter = LVBenchHFDataset(
        split=split,
        video_cache_dir=args.video_cache_dir,
        system_prompt="",
        videomme_use_official_prompt=False,
    )
    backend = HFInProcessBackend(model=model, processor=processor, device=device, max_len=max_len)
    cfg = LoopConfig(
        max_rounds=1,
        max_frames_per_round=int(args.max_frames),
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(args.max_new_tokens),
        request_timeout_s=0,
        max_retries_per_round=0,
        strict_actions=False,
        force_final_answer=False,
        use_candidate_frames=False,
        candidate_k=None,
        use_candidate_frame_ids=False,
        require_candidate_frames=False,
        answer_only_final_round=False,
        observation_mode="image",
        caption_include="none",
        caption_max_chars=0,
        captions_dir=None,
        hide_seen_frames_in_prompt=False,
        log_jsonl=args.log_jsonl or None,
        seed=42 + int(args.shard_idx),
    )

    def _video_path(sample: MCVideoSample) -> str:
        return str(cache_dir / sample.video_key)

    def _log_record(
        sample: MCVideoSample,
        *,
        outcome: Any,
        pred: Optional[str],
        is_correct: bool,
        video_path: str,
        split: str,
    ) -> dict[str, Any]:
        gt = _normalize_answer_letter(sample.answer_letter, len(sample.options))
        return {
            "ts": time.time(),
            "dataset": sample.dataset,
            "split": split,
            "sample_id": sample.sample_id,
            "uid": sample.uid,
            "video_key": sample.video_key,
            "video_url": sample.video_url,
            "video_path": video_path,
            "time_reference": sample.time_reference,
            "frame_indices": list(outcome.frame_indices),
            "question": sample.question,
            "options": sample.options,
            "answer_gt": gt,
            "pred_answer": pred,
            "raw_output": outcome.raw_output,
            "is_correct": bool(is_correct),
        }

    def _build_summary(
        *,
        samples_total: int,
        answered: int,
        correct: int,
        failed: int,
        invalid: int,
        frames_used: int,
        elapsed_s: float,
        stats: RunStats,
    ) -> dict[str, Any]:
        processed = samples_total
        total_rounds = max(0, processed - failed)
        prompt_log_lines = 0
        prompt_log_bytes = 0
        if args.log_jsonl and os.path.exists(args.log_jsonl):
            prompt_log_bytes = os.path.getsize(args.log_jsonl)
            with open(args.log_jsonl, "r", encoding="utf-8") as f:
                prompt_log_lines = sum(1 for _ in f)
        results = {
            "samples": processed,
            "correct": correct,
            "accuracy": correct / max(1, processed),
            "total_rounds": total_rounds,
            "avg_rounds": total_rounds / max(1, processed),
            "total_effective_rounds": total_rounds,
            "avg_effective_rounds": total_rounds / max(1, processed),
            "total_frames_used": frames_used,
            "avg_frames_used": frames_used / max(1, processed),
            "failed": failed,
            "elapsed_s": float(elapsed_s),
            "prompt_log_lines": prompt_log_lines,
            "prompt_log_bytes": prompt_log_bytes,
            "invalid_outputs": invalid,
            "invalid_action_terminated": invalid,
            "total_retries": 0,
            "total_model_calls": stats.total_model_calls,
        }
        wandb_info: Optional[dict[str, Any]] = None
        if run is not None:
            run.summary["final_acc"] = results["accuracy"]
            run.summary["failed"] = failed
            run.summary["invalid_outputs"] = invalid
            run.summary["prompt_log_jsonl"] = args.log_jsonl
            run.summary["prompt_log_lines"] = prompt_log_lines
            run.summary["prompt_log_bytes"] = prompt_log_bytes
            run.finish()
            wandb_info = {
                "enabled": True,
                "mode": getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE"),
                "id": getattr(run, "id", None),
                "url": getattr(run, "url", None),
            }
        return {
            "task": "lvbench_oneshot_hf",
            "dataset": "lvbench",
            "split": args.split,
            "model_path": args.model_path,
            "video_cache_dir": args.video_cache_dir,
            "dtype": args.dtype,
            "max_frames": args.max_frames,
            "max_new_tokens": args.max_new_tokens,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
            "log_jsonl": args.log_jsonl,
            "summary_json": args.summary_json,
            "prompt_log_jsonl": args.log_jsonl,
            "results": results,
            "wandb": wandb_info,
            "command": " ".join(sys.argv),
        }

    args._pnp_setting = "oneshot_baseline"
    args._pnp_split = split
    args._pnp_oneshot_resume_completed = resume_completed
    args._pnp_oneshot_skip_missing_video = True
    args._pnp_oneshot_video_path = _video_path
    args._pnp_oneshot_log_record = _log_record
    args._pnp_oneshot_build_summary = _build_summary
    args._pnp_oneshot_restart_server = None

    stats = RunStats()
    pnp_harness.run_eval(
        samples,
        dataset=dataset_adapter,
        backend=backend,
        cfg=cfg,
        stats=stats,
        rng=rng,
        base_url="",
        model_id=backend.get_model_id("", model_id=args.model_path),
        run=None,
        args=args,
    )


if __name__ == "__main__":
    main()
