#!/usr/bin/env python3

"""Train the NExT-QA Table 4 RL policy and evaluate the saved checkpoint.

The VERL trainer validation path can hold Ray resources after checkpoint save
for this multi-turn video agent. This runner keeps training focused on GRPO
updates, then evaluates the saved policy with the same REVISE/PnP evaluator
used by the plug-and-play Table 4 row so the output is a normal summary JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_command(path: Path, cmd: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(shlex.join([str(part) for part in cmd]) + "\n", encoding="utf-8")


def _write_setting_manifest(path: Path, args: argparse.Namespace, ckpt_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "setting": "nextqa_table4_rl_after_sft",
        "sft_path": args.sft_path,
        "checkpoint_dir": str(ckpt_dir),
        "steps": args.steps,
        "n_gpus": args.n_gpus,
        "rollout_tensor_parallel_size": args.rollout_tensor_parallel_size,
        "eval_tensor_parallel_size": args.eval_tensor_parallel_size,
        "train_batch_size": args.train_batch_size,
        "ppo_mini_batch_size": args.ppo_mini_batch_size,
        "rollout_n": args.rollout_n,
        "max_rounds": args.max_rounds,
        "max_frames_per_round": args.max_frames_per_round,
        "max_retries_per_round": args.max_retries_per_round,
        "min_select_rounds": args.min_select_rounds,
        "use_1fps_timeline": True,
        "reward": {
            "lambda_conf": args.lambda_conf,
            "lambda_sum": args.lambda_sum,
            "lambda_stop": args.lambda_stop,
            "gamma": args.gamma,
            "format_reward": args.format_reward,
            "stop_round_threshold": args.stop_round_threshold,
            "stop_bonus_beta": args.stop_bonus_beta,
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latest_checkpoint(ckpt_dir: Path, fallback_step: int) -> Path:
    tracker = ckpt_dir / "latest_checkpointed_iteration.txt"
    step = int(fallback_step)
    if tracker.is_file():
        raw = tracker.read_text(encoding="utf-8").strip()
        if raw:
            step = int(raw)
    path = ckpt_dir / f"global_step_{step}" / "actor" / "huggingface"
    if not path.is_dir():
        raise FileNotFoundError(f"Expected RL checkpoint directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Expected checkpoint config.json missing: {path / 'config.json'}")
    return path


def _checkpoint_dir(args: argparse.Namespace) -> Path:
    default_ckpt_dir = Path(args.output_dir) / "checkpoints" / "nextqa_table4_grpo_after_sft"
    return Path(args.checkpoint_dir or os.environ.get("REVISE_NEXTQA_RL_CKPT_DIR") or default_ckpt_dir)


def _clear_fresh_eval_outputs(args: argparse.Namespace) -> None:
    for path_value in (args.summary_json, args.log_jsonl):
        if not path_value:
            continue
        try:
            Path(path_value).unlink(missing_ok=True)
        except OSError:
            pass


def _clear_fresh_rl_outputs(args: argparse.Namespace) -> None:
    _clear_fresh_eval_outputs(args)
    if args.skip_train:
        return
    ckpt_dir = _checkpoint_dir(args)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)


def build_train_command(args: argparse.Namespace) -> list[str]:
    ckpt_dir = _checkpoint_dir(args)
    return [
        args.python_bin,
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        str(REPO_ROOT / "revise/config"),
        "--config-name",
        "revise_nextqa_grpo_after_sft",
        "custom_reward_function.path=pkg://verl.utils.reward_score.eager_videoqa_paper",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.model.path={args.sft_path}",
        f"data.nextqa.video_root={args.video_root}",
        f"data.nextqa.map_json={args.map_json}",
        f"data.train_files={args.train_csv}",
        f"data.val_files={args.val_csv}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tensor_parallel_size}",
        "actor_rollout_ref.rollout.data_parallel_size=1",
        "actor_rollout_ref.rollout.max_new_tokens=256",
        f"actor_rollout_ref.rollout.revise.max_rounds={args.max_rounds}",
        f"actor_rollout_ref.rollout.revise.max_frames_per_round={args.max_frames_per_round}",
        f"actor_rollout_ref.rollout.revise.max_retries_per_round={args.max_retries_per_round}",
        f"actor_rollout_ref.rollout.revise.min_select_rounds={args.min_select_rounds}",
        "actor_rollout_ref.rollout.revise.use_1fps_timeline=True",
        "actor_rollout_ref.rollout.revise.compute_margins=True",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"custom_reward_function.reward_kwargs.lambda_conf={args.lambda_conf}",
        f"custom_reward_function.reward_kwargs.lambda_sum={args.lambda_sum}",
        f"custom_reward_function.reward_kwargs.lambda_stop={args.lambda_stop}",
        f"custom_reward_function.reward_kwargs.gamma={args.gamma}",
        f"custom_reward_function.reward_kwargs.format_reward={args.format_reward}",
        f"custom_reward_function.reward_kwargs.stop_round_threshold={args.stop_round_threshold}",
        f"custom_reward_function.reward_kwargs.stop_bonus_beta={args.stop_bonus_beta}",
        f"data.train_batch_size={args.train_batch_size}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"trainer.total_training_steps={args.steps}",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        f"trainer.save_freq={args.steps}",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.logger={args.logger_json}",
    ]


def build_eval_command(args: argparse.Namespace, checkpoint_path: Path) -> list[str]:
    return [
        args.python_bin,
        str(REPO_ROOT / "revise/benchmarks/nextqa_vllm.py"),
        "--video-root",
        args.video_root,
        "--map-json",
        args.map_json,
        "--csv",
        args.val_csv,
        "--seed",
        "0",
        "--max-rounds",
        str(args.max_rounds),
        "--max-frames-per-round",
        str(args.max_frames_per_round),
        "--max-retries-per-round",
        str(args.max_retries_per_round),
        "--min-select-rounds",
        str(args.min_select_rounds),
        "--max-samples",
        str(args.max_samples),
        "--summary-json",
        args.summary_json,
        "--log-jsonl",
        args.log_jsonl,
        "--no-resume-from-log",
        "--model-path",
        str(checkpoint_path),
        "--start-server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.eval_tensor_parallel_size),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--server-timeout-s",
        str(args.server_timeout_s),
        "--server-log",
        args.server_log,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--map-json", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--sft-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Directory for large VERL checkpoints. Defaults to "
            "$REVISE_NEXTQA_RL_CKPT_DIR or <output-dir>/checkpoints/nextqa_table4_grpo_after_sft."
        ),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=8)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--lambda-sum", type=float, default=1.0)
    parser.add_argument("--lambda-stop", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--format-reward", type=float, default=0.05)
    parser.add_argument("--stop-round-threshold", type=int, default=2)
    parser.add_argument("--stop-bonus-beta", type=float, default=0.0)
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--rollout-tensor-parallel-size", type=int, default=4)
    parser.add_argument("--eval-tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-frames-per-round", type=int, default=3)
    parser.add_argument("--max-retries-per-round", type=int, default=1)
    parser.add_argument("--min-select-rounds", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--log-jsonl", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--logger-json", default='["console","wandb"]')
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--server-timeout-s", type=int, default=1800)
    parser.add_argument("--skip-train", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _checkpoint_dir(args)
    _clear_fresh_rl_outputs(args)

    train_cmd = build_train_command(args)
    _write_command(out_dir / "nextqa_table4_rl_after_sft.train.cmd", train_cmd)
    _write_setting_manifest(out_dir / "nextqa_table4_rl_after_sft.settings.json", args, ckpt_dir)
    if not args.skip_train:
        proc = subprocess.run(train_cmd, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            return int(proc.returncode)

    checkpoint_path = _latest_checkpoint(ckpt_dir, args.steps)
    eval_cmd = build_eval_command(args, checkpoint_path)
    _write_command(out_dir / "nextqa_table4_rl_after_sft.eval.cmd", eval_cmd)
    proc = subprocess.run(eval_cmd, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        return int(proc.returncode)

    summary_path = Path(args.summary_json)
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
        if isinstance(summary, dict):
            summary.setdefault("checkpoint_path", str(checkpoint_path))
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
