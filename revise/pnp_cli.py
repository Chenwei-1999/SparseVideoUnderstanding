"""Unified CLI for multi-round plug-and-play REVISE evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise.pnp import harness as pnp_harness
from revise.pnp.protocols import LoopConfig, RunStats
from revise.pnp.utils import (
    get_model_id,
    maybe_init_wandb,
    pick_free_port,
    resolve_base_url,
    shard_by_video,
    wait_for_server,
    wandb_log,
)

DATASET_NAMES = ("egoschema", "lvbench", "nextqa", "videoespresso", "videomme")
LEGACY_DATASET_ALIASES: dict[str, str] = {}
BACKEND_NAMES = ("hf_inprocess", "vllm_http")
LOCAL_MC_DATASETS = {"egoschema", "nextqa", "videoespresso"}
LONG_VIDEO_DATASETS = {"lvbench", "videomme"}


def _resolve(kind: str, name: str) -> type[Any]:
    from revise.pnp.registry import resolve

    return resolve(kind, name)


def _dataset_name(value: str) -> str:
    name = str(value).strip().lower()
    if name in DATASET_NAMES or name in LEGACY_DATASET_ALIASES:
        return name
    expected = ", ".join((*DATASET_NAMES, *LEGACY_DATASET_ALIASES))
    raise argparse.ArgumentTypeError(f"unknown dataset {value!r}; expected one of: {expected}")


def _normalize_legacy_dataset_alias(args: argparse.Namespace) -> None:
    if str(args.dataset).lower() in LEGACY_DATASET_ALIASES:
        if args.backend != "hf_inprocess":
            raise ValueError(f"--dataset {args.dataset} is a legacy alias and requires --backend hf_inprocess.")
        args._legacy_dataset_alias = str(args.dataset).lower()
        args.dataset = LEGACY_DATASET_ALIASES[args._legacy_dataset_alias]


def _apply_setting_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.setting != "oneshot_baseline":
        return
    if bool(getattr(args, "preserve_setting_defaults", False)):
        return
    if args.temperature == parser.get_default("temperature"):
        args.temperature = 0.0
    if args.top_p == parser.get_default("top_p"):
        args.top_p = 1.0
    if args.max_tokens == parser.get_default("max_tokens"):
        args.max_tokens = 16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified REVISE plug-and-play evaluator.")
    parser.add_argument(
        "--dataset",
        type=_dataset_name,
        metavar="{" + ",".join(DATASET_NAMES) + "}",
        required=True,
    )
    parser.add_argument("--backend", choices=BACKEND_NAMES, required=True)
    parser.add_argument(
        "--setting",
        default="multi_round_pnp",
        choices=["multi_round_pnp", "oneshot_baseline"],
    )

    parser.add_argument("--model-path", default="")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--map-json", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--split", default="")
    parser.add_argument("--video-cache-dir", default="./data/revise_assets/video_cache")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=0)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--restart-server-on-failure", action="store_true")
    parser.add_argument("--server-log", default="")
    parser.add_argument("--server-timeout-s", type=int, default=300)
    parser.add_argument("--request-timeout-s", "--timeout-s", dest="request_timeout_s", type=int, default=300)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)

    # Paper setting (Sec. 4 "Settings"): max_rounds=4 (T=4), max_frames_per_round=3.
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-frames-per-round", "--max-frames", type=int, default=3)
    parser.add_argument(
        "--min-select-rounds",
        type=int,
        default=0,
        help=(
            "Require this many successful <select> rounds before accepting <answer>; "
            "default 0 preserves paper early stopping. Positive values are diagnostic ablations only."
        ),
    )
    parser.add_argument("--candidate-k", type=int, default=0)
    parser.add_argument("--use-candidate-frames", action="store_true")
    parser.add_argument("--use-candidate-frame-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-candidate-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hide-seen-frames-in-prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-retries-per-round", type=int, default=0)
    parser.add_argument("--strict-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=256)
    parser.add_argument(
        "--preserve-setting-defaults",
        action="store_true",
        help="Do not apply setting-specific decoding overrides; used for paper-comparable one-shot rows.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-final-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--answer-only-final-round", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fallback-on-invalid-candidate-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-mode", choices=["image", "caption"], default="image")
    parser.add_argument("--captions-dir", default=None)
    parser.add_argument("--caption-include", choices=["none", "shown", "candidate", "both"], default="none")
    parser.add_argument("--caption-max-chars", type=int, default=200)

    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--log-jsonl", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--resume-from-log", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ablate-state-carryover", action="store_true")
    parser.add_argument("--ablate-structured-summary", action="store_true")
    parser.add_argument("--videoespresso-use-official-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--videoespresso-with-evidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--videomme-use-official-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yt-dlp-timeout-s", type=int, default=600)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--allow-missing-cached-videos", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "verl-revise"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS", ""))
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", ""))
    return parser


def _slice_samples(samples: list[Any], args: argparse.Namespace) -> list[Any]:
    start_idx = max(0, int(args.start_idx or 0))
    end_idx = int(args.end_idx or 0)
    if end_idx <= 0:
        end_idx = len(samples)
    samples = samples[start_idx:end_idx]
    if int(args.max_samples or 0) > 0:
        samples = samples[: int(args.max_samples)]
    return samples


def _suffix_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"


def _suffix_sharded_outputs(args: argparse.Namespace) -> None:
    num_shards = max(1, int(args.num_shards or 1))
    if num_shards <= 1:
        return
    suffix = f".shard{int(args.shard_idx)}of{num_shards}"
    if args.log_jsonl and suffix not in args.log_jsonl:
        args.log_jsonl = _suffix_path(args.log_jsonl, suffix)
    if args.summary_json and suffix not in args.summary_json:
        args.summary_json = _suffix_path(args.summary_json, suffix)


def _clear_outputs_for_fresh_run(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resume_from_log", False)):
        return
    for attr in ("log_jsonl", "summary_json"):
        path = getattr(args, attr, "") or ""
        if not path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def _completed_ids_from_log(path: str) -> set[str]:
    completed: set[str] = set()
    if not path or not os.path.exists(path):
        return completed
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("sample_id")
            if sid and obj.get("pred_answer"):
                completed.add(str(sid))
    return completed


def _filter_cached_long_video_samples(args: argparse.Namespace, samples: list[Any]) -> list[Any]:
    cache_dir = Path(args.video_cache_dir) / str(args.dataset).lower()
    args._pnp_oneshot_cache_dir = cache_dir
    args._pnp_cache_filter_total = len(samples)
    args._pnp_cache_filter_missing = 0
    args._pnp_cache_filter_missing_examples = []

    if not bool(args.cached_only):
        return samples

    filtered: list[Any] = []
    missing_keys: list[str] = []
    for sample in samples:
        path = cache_dir / sample.video_key
        try:
            ok = path.stat().st_size > 0
        except FileNotFoundError:
            ok = False
        if ok:
            filtered.append(sample)
        else:
            missing_keys.append(sample.video_key)

    unique_missing = set(missing_keys)
    args._pnp_cache_filter_missing = len(unique_missing)
    args._pnp_cache_filter_missing_examples = sorted(unique_missing)[:20]
    if args._pnp_cache_filter_missing and int(args.max_samples or 0) <= 0 and not args.allow_missing_cached_videos:
        total_unique = len({s.video_key for s in samples})
        raise SystemExit(
            f"--cached-only missing {args._pnp_cache_filter_missing} distinct {args.dataset} videos "
            f"out of {total_unique}; examples={args._pnp_cache_filter_missing_examples}. "
            "Populate the cache or pass --allow-missing-cached-videos for an explicitly partial run."
        )
    return filtered


def _select_samples(args: argparse.Namespace, samples: list[Any]) -> list[Any]:
    dataset_name = str(args.dataset).lower()
    if dataset_name in LONG_VIDEO_DATASETS and args.backend == "vllm_http":
        samples = _filter_cached_long_video_samples(args, samples)
    samples = _slice_samples(samples, args)

    if args.setting != "oneshot_baseline":
        return samples

    _suffix_sharded_outputs(args)
    if dataset_name in LOCAL_MC_DATASETS:
        samples = shard_by_video(samples, args.num_shards, args.shard_idx, video_key_attr="video_path")
    elif dataset_name in LONG_VIDEO_DATASETS:
        samples = shard_by_video(samples, args.num_shards, args.shard_idx)
        samples.sort(key=lambda s: (s.video_key, s.uid))

    if args.resume_from_log and args.log_jsonl:
        completed = _completed_ids_from_log(args.log_jsonl)
        if completed:
            args._pnp_oneshot_resume_completed = len(completed)
            samples = [s for s in samples if str(getattr(s, "sample_id", "")) not in completed]

    return samples


def _load_samples(args: argparse.Namespace) -> tuple[list[Any], str]:
    dataset_name = str(args.dataset).lower()
    if dataset_name == "nextqa":
        if not (args.csv and args.map_json and args.video_root):
            raise ValueError("--dataset nextqa requires --csv, --map-json, and --video-root.")
        from revise.datasets.nextqa import load_samples

        samples = load_samples(
            csv_path=args.csv,
            map_json=args.map_json,
            video_root=args.video_root,
            max_samples=0 if args.setting == "oneshot_baseline" else args.max_samples,
            seed=args.seed or 42,
        )
        return samples, ""

    if dataset_name in {"egoschema", "videoespresso"}:
        if not (args.json and args.video_root):
            raise ValueError(f"--dataset {dataset_name} requires --json and --video-root.")
        from revise.datasets.egoschema import load_samples

        samples = load_samples(
            args.json,
            args.video_root,
            0 if args.setting == "oneshot_baseline" else args.max_samples,
            args.seed,
            allow_video_download=False,
            egoschema_video_repo="VLM2Vec/egoschema-rawvideo",
        )
        args.dataset_name = dataset_name
        return samples, ""

    if dataset_name == "videomme":
        split = args.split or "test"
        if args.backend == "hf_inprocess":
            from revise.datasets.videomme import load_hf_samples as load_samples
        else:
            from revise.datasets.videomme import load_samples

        samples = load_samples(split)
        return samples, split

    if dataset_name == "lvbench":
        split = args.split or "train"
        if args.backend == "hf_inprocess":
            from revise.datasets.lvbench import load_hf_samples

            samples = load_hf_samples(split)
        else:
            from revise.datasets.lvbench import load_samples

            samples = load_samples(split)
        return samples, split

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _build_dataset(args: argparse.Namespace, split: str) -> Any:
    dataset_name = str(args.dataset).lower()
    if args.backend == "hf_inprocess":
        from revise.datasets.lvbench import DEFAULT_CACHED_LONG_VIDEO_SYSTEM_PROMPT, CachedLongVideoDataset

        dataset_cls = CachedLongVideoDataset
    else:
        dataset_key = (
            "local_mc" if args.setting == "oneshot_baseline" and dataset_name in LOCAL_MC_DATASETS else dataset_name
        )
        dataset_cls = _resolve("dataset", dataset_key)
    if dataset_name == "nextqa":
        if args.setting == "oneshot_baseline":
            return dataset_cls(
                dataset_name="nextqa",
                videoespresso_use_official_prompt=False,
                videoespresso_with_evidence=False,
            )
        return dataset_cls()
    if dataset_name in {"egoschema", "videoespresso"}:
        if args.setting == "oneshot_baseline":
            return dataset_cls(
                dataset_name=dataset_name,
                videoespresso_use_official_prompt=bool(args.videoespresso_use_official_prompt),
                videoespresso_with_evidence=bool(args.videoespresso_with_evidence),
            )
        return dataset_cls(
            dataset_name=dataset_name,
            structured_summary=not bool(args.ablate_structured_summary),
            carry_summary_state=not bool(args.ablate_state_carryover),
            videoespresso_use_official_prompt=bool(args.videoespresso_use_official_prompt),
            videoespresso_with_evidence=bool(args.videoespresso_with_evidence),
        )
    if args.backend == "hf_inprocess":
        return dataset_cls(
            split=split,
            video_cache_dir=args.video_cache_dir,
            system_prompt=DEFAULT_CACHED_LONG_VIDEO_SYSTEM_PROMPT,
            videomme_use_official_prompt=bool(args.videomme_use_official_prompt),
        )
    return dataset_cls(
        split=split,
        video_cache_dir=args.video_cache_dir,
        yt_dlp_timeout_s=args.yt_dlp_timeout_s,
        videomme_use_official_prompt=bool(args.videomme_use_official_prompt),
    )


def _build_loop_config(args: argparse.Namespace) -> LoopConfig:
    dataset_name = str(args.dataset).lower()
    if args.setting == "oneshot_baseline":
        return LoopConfig(
            max_rounds=1,
            max_frames_per_round=int(args.max_frames_per_round),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_tokens),
            request_timeout_s=int(args.request_timeout_s),
            max_retries_per_round=0,
            min_select_rounds=0,
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
            seed=int(args.seed),
        )
    if dataset_name in {"videomme", "lvbench"}:
        args.use_candidate_frames = True
        if args.candidate_k <= 0:
            args.candidate_k = 20
        args.use_candidate_frame_ids = True
        args.require_candidate_frames = True
        strict_actions = False
        fallback = True
    elif dataset_name in {"egoschema", "videoespresso"}:
        if args.candidate_k <= 0:
            args.candidate_k = max(12, args.max_frames_per_round * 4)
        strict_actions = bool(args.strict_actions)
        fallback = False
    else:
        if args.use_candidate_frame_ids or args.require_candidate_frames:
            args.use_candidate_frames = True
        strict_actions = bool(args.strict_actions)
        fallback = bool(args.fallback_on_invalid_candidate_ids)
        if args.candidate_k <= 0:
            args.candidate_k = None
    return LoopConfig(
        max_rounds=args.max_rounds,
        max_frames_per_round=args.max_frames_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        request_timeout_s=args.request_timeout_s,
        max_retries_per_round=args.max_retries_per_round,
        min_select_rounds=max(0, int(getattr(args, "min_select_rounds", 0) or 0)),
        strict_actions=strict_actions,
        force_final_answer=bool(args.force_final_answer),
        use_candidate_frames=bool(args.use_candidate_frames),
        candidate_k=args.candidate_k,
        use_candidate_frame_ids=bool(args.use_candidate_frame_ids),
        require_candidate_frames=bool(args.require_candidate_frames),
        answer_only_final_round=bool(args.answer_only_final_round),
        observation_mode=getattr(args, "observation_mode", "image"),
        caption_include=getattr(args, "caption_include", "none"),
        caption_max_chars=int(getattr(args, "caption_max_chars", 0)),
        captions_dir=getattr(args, "captions_dir", None),
        hide_seen_frames_in_prompt=bool(getattr(args, "hide_seen_frames_in_prompt", False)),
        log_jsonl=args.log_jsonl or None,
        seed=int(args.seed),
        fallback_on_invalid_candidate_ids=fallback,
    )


def _vllm_image_resize_limit(args: argparse.Namespace) -> int:
    """Return max image edge for vLLM HTTP payloads.

    Local MC benchmarks use a small number of decoded frames and historically
    sent them without resizing. Long-video benchmarks keep a resize cap to avoid
    large OpenAI-compatible request payloads.
    """
    dataset_name = str(args.dataset).lower()
    if dataset_name in LOCAL_MC_DATASETS and getattr(args, "observation_mode", "image") == "image":
        return 0
    return 384


def _assign_vllm_port(args: argparse.Namespace) -> None:
    """Fill the HTTP port without opening sockets unless we launch vLLM here."""
    if args.backend != "vllm_http" or args.port > 0:
        return
    if args.start_server:
        args.port = pick_free_port()
    elif not args.base_url:
        args.port = 8000


def _prepare_oneshot_metadata(args: argparse.Namespace, split: str, dataset_adapter: Any) -> None:
    """Wire the shared single-round baseline harness path."""
    dataset_name = str(args.dataset).lower()
    args._pnp_setting = "oneshot_baseline"
    args._pnp_split = split
    args._pnp_oneshot_resume_completed = int(getattr(args, "_pnp_oneshot_resume_completed", 0) or 0)
    args._pnp_oneshot_cache_dir = getattr(
        args,
        "_pnp_oneshot_cache_dir",
        Path(args.video_cache_dir) / dataset_name,
    )
    args._pnp_oneshot_cache_filter_total = int(getattr(args, "_pnp_cache_filter_total", 0) or 0)
    args._pnp_oneshot_cache_filter_missing = int(getattr(args, "_pnp_cache_filter_missing", 0) or 0)
    args._pnp_oneshot_cache_filter_missing_examples = list(
        getattr(args, "_pnp_cache_filter_missing_examples", []) or []
    )
    args._pnp_oneshot_skip_missing_video = True
    args._pnp_oneshot_restart_server = None

    if dataset_name not in LOCAL_MC_DATASETS:
        return

    args._pnp_oneshot_skip_missing_video = False
    args._pnp_oneshot_video_path = lambda sample: sample.video_path

    is_videoespresso_official = bool(dataset_name == "videoespresso" and args.videoespresso_use_official_prompt)
    is_videoespresso_evidence = bool(dataset_name == "videoespresso" and args.videoespresso_with_evidence)

    def _log_record(
        sample: Any,
        *,
        outcome: Any,
        pred: str | None,
        is_correct: bool,
        video_path: str,
        split: str,
    ) -> dict[str, Any]:
        _ = split, video_path
        frame_indices = list(outcome.frame_indices)
        question_block = dataset_adapter.format_question(sample)
        from revise.datasets.local_mc import build_oneshot_user_text

        user_text = build_oneshot_user_text(question_block, frame_indices)
        gt = dataset_adapter.ground_truth_letter(sample)
        return {
            "ts": time.time(),
            "dataset": dataset_name,
            "sample_id": sample.sample_id,
            "qid": getattr(sample, "qid", ""),
            "video_path": sample.video_path,
            "question": sample.question,
            "options": sample.choices,
            "task": getattr(sample, "task", ""),
            "evidence": getattr(sample, "evidence", ""),
            "system_prompt": "",
            "user_text": user_text,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": outcome.raw_output},
            ],
            "frame_indices": frame_indices,
            "pred_answer": pred,
            "answer_gt": gt,
            "correct": bool(is_correct),
            "raw_output": outcome.raw_output,
            "videoespresso_official_prompt": is_videoespresso_official,
            "videoespresso_with_evidence": is_videoespresso_evidence,
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
        _ = invalid
        return {
            "task": "oneshot_local_mc",
            "dataset": dataset_name,
            "samples": samples_total,
            "answered": answered,
            "correct": correct,
            "accuracy": float(correct / max(1, answered)),
            "failed": failed,
            "avg_frames": float(frames_used / max(1, answered)),
            "elapsed_s": float(elapsed_s),
            "total_model_calls": stats.total_model_calls,
            "log_jsonl": args.log_jsonl,
            "videoespresso_use_official_prompt": bool(args.videoespresso_use_official_prompt)
            if dataset_name == "videoespresso"
            else None,
            "videoespresso_with_evidence": bool(args.videoespresso_with_evidence)
            if dataset_name == "videoespresso"
            else None,
        }

    args._pnp_oneshot_log_record = _log_record
    args._pnp_oneshot_build_summary = _build_summary


def _prepare_harness_metadata(
    args: argparse.Namespace,
    split: str,
    max_len: int | None = None,
    dataset_adapter: Any | None = None,
) -> None:
    dataset_name = str(args.dataset).lower()
    args._pnp_maybe_init_wandb = maybe_init_wandb
    args._pnp_wandb_log = wandb_log
    args._pnp_split = split
    args._pnp_max_len = max_len
    if getattr(args, "setting", "multi_round_pnp") == "oneshot_baseline":
        _prepare_oneshot_metadata(args, split, dataset_adapter)
        return
    if dataset_name == "nextqa":
        from revise.datasets.nextqa import load_progress_from_log

        args._pnp_harness_mode = "nextqa"
        args._pnp_load_progress_from_log = load_progress_from_log
        args._pnp_run_config = {
            "task": "revise_plug_and_play_nextqa_vllm",
            "dataset_csv": args.csv,
            "video_root": args.video_root,
            "map_json": args.map_json,
            "model_path": args.model_path,
            "engine": "vllm",
            "max_samples": args.max_samples,
            "max_rounds": args.max_rounds,
            "max_frames_per_round": args.max_frames_per_round,
            "max_retries_per_round": args.max_retries_per_round,
            "min_select_rounds": args.min_select_rounds,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
        }
        args._pnp_summary_payload = dict(args._pnp_run_config)
    elif dataset_name in {"egoschema", "videoespresso"}:
        args._pnp_harness_mode = "egoschema"
        args._pnp_summary_payload = {
            "task": f"revise_plug_and_play_{dataset_name}_vllm",
            "dataset_name": dataset_name,
            "dataset_json": args.json,
            "video_root": args.video_root,
            "model_path": args.model_path,
            "engine": "vllm",
            "max_samples": args.max_samples,
            "max_rounds": args.max_rounds,
            "max_frames_per_round": args.max_frames_per_round,
            "max_retries_per_round": args.max_retries_per_round,
            "min_select_rounds": args.min_select_rounds,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
        }
    elif args.backend == "hf_inprocess":
        args._pnp_harness_mode = "long_hf"
        args._pnp_run_config = {
            "task": "revise_pnp_hf_inprocess",
            "dataset": args.dataset,
            "split": split,
            "model_path": args.model_path,
            "max_rounds": args.max_rounds,
            "max_frames_per_round": args.max_frames_per_round,
            "min_select_rounds": args.min_select_rounds,
            "candidate_k": args.candidate_k,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_tokens,
        }
    else:
        args._pnp_harness_mode = "long_vllm"
        args._pnp_run_config = {
            "task": "revise_plug_and_play_videomme_lvbench_vllm",
            "dataset": args.dataset,
            "split": split,
            "model_path": args.model_path,
            "video_cache_dir": args.video_cache_dir,
            "max_samples": args.max_samples,
            "max_rounds": args.max_rounds,
            "max_frames_per_round": args.max_frames_per_round,
            "max_retries_per_round": args.max_retries_per_round,
            "min_select_rounds": args.min_select_rounds,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
        }


def main(argv: list[str] | None = None) -> int | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _normalize_legacy_dataset_alias(args)
    _apply_setting_defaults(args, parser)
    _clear_outputs_for_fresh_run(args)
    if args.backend == "hf_inprocess" and args.dataset not in LONG_VIDEO_DATASETS:
        raise ValueError("--backend hf_inprocess is supported with long-video datasets: lvbench or videomme.")
    if args.base_url and args.start_server:
        raise ValueError("--base-url cannot be combined with --start-server.")
    _assign_vllm_port(args)

    samples, split = _load_samples(args)
    samples = _select_samples(args, samples)
    _suffix_sharded_outputs(args)
    _clear_outputs_for_fresh_run(args)
    if not samples:
        raise RuntimeError("No samples loaded.")

    random.seed(args.seed)
    rng = random.Random(args.seed)
    dataset = _build_dataset(args, split)
    cfg = _build_loop_config(args)
    stats = RunStats()

    server_proc = None
    base_url = ""
    model_id = ""
    backend: Any
    max_len: int | None = None
    try:
        if args.backend == "vllm_http":
            backend_cls = _resolve("backend", "vllm_http")
            max_edge = _vllm_image_resize_limit(args)
            backend = backend_cls(max_edge=max_edge, quality=90 if max_edge <= 0 else 85)
            if args.start_server:
                from revise.pnp.utils import start_vllm_server

                cuda_default = "0" if str(args.dataset).lower() in LONG_VIDEO_DATASETS else "0,1,2,3"
                server_proc = start_vllm_server(
                    args,
                    image_limit=int(args.max_frames_per_round),
                    cuda_visible_default=cuda_default,
                )
                wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
            base_url = resolve_base_url(args.base_url, args.host, args.port)
            model_id = get_model_id(base_url, model_id=args.model_id)
        else:
            import torch

            from revise.backends.hf_inprocess import _load_model_and_processor

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model, processor = _load_model_and_processor(args.model_path, args.dtype, device)
            max_len = int(getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", 32768))
            backend_cls = _resolve("backend", "hf_inprocess")
            backend = backend_cls(model=model, processor=processor, device=device, max_len=max_len)
            model_id = backend.get_model_id("", model_id=args.model_path)

        _prepare_harness_metadata(args, split, max_len=max_len, dataset_adapter=dataset)
        result = pnp_harness.run_eval(
            samples,
            dataset=dataset,
            backend=backend,
            cfg=cfg,
            stats=stats,
            rng=rng,
            base_url=base_url,
            model_id=model_id,
            run=None,
            args=args,
        )
        if getattr(args, "_pnp_harness_mode", "") in {"egoschema", "long_vllm"}:
            if result.get("samples", 0) > 0 and (
                result.get("failed", 0) >= result.get("samples", 0) or result.get("total_model_calls", 0) == 0
            ):
                return 2
        return None if getattr(args, "_pnp_harness_mode", "") == "long_hf" else 0
    finally:
        if server_proc is not None:
            from revise.pnp.utils import stop_server

            stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
