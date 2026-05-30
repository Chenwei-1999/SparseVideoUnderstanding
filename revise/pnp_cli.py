"""Unified CLI for multi-round plug-and-play REVISE evaluation."""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

from revise.pnp_protocols import LoopConfig, RunStats
from revise.pnp_utils import (
    get_model_id,
    maybe_init_wandb,
    pick_free_port,
    resolve_base_url,
    wandb_log,
    wait_for_server,
)
from revise import pnp_harness

DATASET_NAMES = ("egoschema", "lvbench", "lvbench_hf", "nextqa", "videoespresso", "videomme")
BACKEND_NAMES = ("hf_inprocess", "vllm_http")


def _resolve(kind: str, name: str) -> type[Any]:
    from revise.pnp_registry import resolve

    return resolve(kind, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified REVISE plug-and-play evaluator.")
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--backend", choices=BACKEND_NAMES, required=True)
    parser.add_argument("--setting", default="multi_round_pnp", choices=["multi_round_pnp"])

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

    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-frames-per-round", "--max-frames", type=int, default=5)
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
    parser.add_argument("--system-prompt", default="")

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


def _load_samples(args: argparse.Namespace) -> tuple[list[Any], str]:
    dataset_name = str(args.dataset).lower()
    if dataset_name == "nextqa":
        if not (args.csv and args.map_json and args.video_root):
            raise ValueError("--dataset nextqa requires --csv, --map-json, and --video-root.")
        from revise import plug_and_play_nextqa_vllm as nextqa

        samples = nextqa._load_nextqa_samples(
            csv_path=args.csv,
            map_json=args.map_json,
            video_root=args.video_root,
            max_samples=args.max_samples,
            seed=args.seed or 42,
        )
        return samples, ""

    if dataset_name in {"egoschema", "videoespresso"}:
        if not (args.json and args.video_root):
            raise ValueError(f"--dataset {dataset_name} requires --json and --video-root.")
        from revise import plug_and_play_egoschema_vllm as egoschema

        samples = egoschema._load_egoschema_samples(
            args.json,
            args.video_root,
            args.max_samples,
            args.seed,
            allow_video_download=False,
            egoschema_video_repo="VLM2Vec/egoschema-rawvideo",
        )
        args.dataset_name = dataset_name
        return samples, ""

    if dataset_name in {"videomme", "lvbench"}:
        from revise import plug_and_play_videomme_lvbench_vllm as long_vllm

        split = args.split or ("test" if dataset_name == "videomme" else "train")
        samples = long_vllm._load_videomme_samples(split) if dataset_name == "videomme" else long_vllm._load_lvbench_samples(split)
        samples = _slice_samples(samples, args)
        return samples, split

    if dataset_name == "lvbench_hf":
        from revise import plug_and_play_lvbench_hf as lvbench_hf

        split = args.split or "train"
        samples = _slice_samples(lvbench_hf._load_lvbench_samples(split), args)
        args.dataset = "lvbench"
        return samples, split

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _build_dataset(args: argparse.Namespace, split: str) -> Any:
    dataset_name = str(args.dataset).lower()
    dataset_cls = _resolve("dataset", "lvbench_hf" if dataset_name == "lvbench" and args.backend == "hf_inprocess" else dataset_name)
    if dataset_name == "nextqa":
        return dataset_cls()
    if dataset_name in {"egoschema", "videoespresso"}:
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
            system_prompt=str(args.system_prompt or ""),
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
    if dataset_name in {"videomme", "lvbench"}:
        args.use_candidate_frames = True
        if args.candidate_k <= 0:
            args.candidate_k = 20
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


def _prepare_harness_metadata(args: argparse.Namespace, split: str, max_len: int | None = None) -> None:
    dataset_name = str(args.dataset).lower()
    args._pnp_maybe_init_wandb = maybe_init_wandb
    args._pnp_wandb_log = wandb_log
    args._pnp_split = split
    args._pnp_max_len = max_len
    if dataset_name == "nextqa":
        args._pnp_harness_mode = "nextqa"
        args._pnp_run_config = {
            "task": "revise_plug_and_play_nextqa_vllm",
            "dataset_csv": args.csv,
            "video_root": args.video_root,
            "map_json": args.map_json,
            "model_path": args.model_path,
            "engine": "vllm",
            "max_samples": args.max_samples,
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
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
        }
    elif args.backend == "hf_inprocess":
        args._pnp_harness_mode = "lvbench_hf"
        args._pnp_run_config = {
            "task": "revise_plug_and_play_lvbench_hf",
            "dataset": args.dataset,
            "split": split,
            "model_path": args.model_path,
            "max_rounds": args.max_rounds,
            "max_frames_per_round": args.max_frames_per_round,
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
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
        }


def main(argv: list[str] | None = None) -> int | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend == "hf_inprocess" and args.dataset != "lvbench_hf":
        raise ValueError("--backend hf_inprocess is currently supported with --dataset lvbench_hf.")
    if args.backend == "vllm_http" and args.dataset == "lvbench_hf":
        raise ValueError("--dataset lvbench_hf requires --backend hf_inprocess.")
    if args.base_url and args.start_server:
        raise ValueError("--base-url cannot be combined with --start-server.")
    if args.port <= 0:
        args.port = pick_free_port() if args.backend == "vllm_http" else 0

    samples, split = _load_samples(args)
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
            backend = backend_cls()
            if args.start_server:
                from revise.pnp_utils import start_vllm_server

                server_proc = start_vllm_server(args)
                wait_for_server(args.host, args.port, timeout_s=args.server_timeout_s)
            base_url = resolve_base_url(args.base_url, args.host, args.port)
            model_id = get_model_id(base_url, model_id=args.model_id)
        else:
            import torch
            from revise.plug_and_play_lvbench_hf import _load_model_and_processor

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model, processor = _load_model_and_processor(args.model_path, args.dtype, device)
            max_len = int(getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", 32768))
            backend_cls = _resolve("backend", "hf_inprocess")
            backend = backend_cls(model=model, processor=processor, device=device, max_len=max_len)
            model_id = backend.get_model_id("", model_id=args.model_path)

        _prepare_harness_metadata(args, split, max_len=max_len)
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
        if args._pnp_harness_mode in {"egoschema", "long_vllm"}:
            if result.get("samples", 0) > 0 and (
                result.get("failed", 0) >= result.get("samples", 0) or result.get("total_model_calls", 0) == 0
            ):
                return 2
        return None if args._pnp_harness_mode == "lvbench_hf" else 0
    finally:
        if server_proc is not None:
            from revise.pnp_utils import stop_server

            stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
