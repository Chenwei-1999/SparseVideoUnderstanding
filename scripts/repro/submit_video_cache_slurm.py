#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.revise.plug_and_play_videomme_lvbench_vllm import (  # noqa: E402
    _load_lvbench_samples,
    _load_videomme_samples,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _distinct_video_count(dataset: str, split: str) -> int:
    samples = _load_videomme_samples(split) if dataset == "videomme" else _load_lvbench_samples(split)
    return len({sample.video_key for sample in samples})


def _sbatch_script(
    *,
    dataset: str,
    split: str,
    start_idx: int,
    max_videos: int,
    manifest_path: Path,
    video_cache_dir: Path,
    conda_sh: str,
    conda_env: str,
    partition: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    account: str,
    yt_dlp_timeout_s: int,
    logs_dir: Path,
) -> str:
    account_line = f"#SBATCH --account={account}\n" if account else ""
    job_name = f"revise-cache-{dataset}-{start_idx}"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{account_line}#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={logs_dir}/%x-%j.out
#SBATCH --error={logs_dir}/%x-%j.err

set -euo pipefail

cd {shlex.quote(str(REPO_ROOT))}
source {shlex.quote(conda_sh)}
conda activate {shlex.quote(conda_env)}

export PYTHONUNBUFFERED=1
export HF_HOME="${{REVISE_HF_HOME:-{shlex.quote(str(REPO_ROOT / "data" / "revise_assets" / ".hf_home"))}}}"
export HF_HUB_CACHE="${{HF_HUB_CACHE:-$HF_HOME/hub}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-$HF_HOME/datasets}}"
export HF_XET_CACHE="${{HF_XET_CACHE:-$HF_HOME/xet}}"

python scripts/repro/cache_hf_video_benchmark.py \\
  --dataset {shlex.quote(dataset)} \\
  --split {shlex.quote(split)} \\
  --video-cache-dir {shlex.quote(str(video_cache_dir))} \\
  --start-idx {start_idx} \\
  --max-videos {max_videos} \\
  --yt-dlp-timeout-s {yt_dlp_timeout_s} \\
  --manifest {shlex.quote(str(manifest_path))}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and optionally submit Slurm video-cache jobs.")
    parser.add_argument(
        "--dataset",
        choices=["videomme", "lvbench"],
        action="append",
        help="Defaults to lvbench. Use submit_videomme_official_download_slurm.py for full Video-MME.",
    )
    parser.add_argument("--split", default="", help="Defaults to Video-MME test or LVBench train per dataset.")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "repro_runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--video-cache-dir",
        default=str(REPO_ROOT / "data" / "revise_assets" / "video_cache"),
    )
    parser.add_argument("--partition", default=os.getenv("REVISE_CACHE_SLURM_PARTITION", "normal"))
    parser.add_argument("--cpus-per-task", type=int, default=int(os.getenv("REVISE_CACHE_SLURM_CPUS", "2")))
    parser.add_argument("--mem", default=os.getenv("REVISE_CACHE_SLURM_MEM", "8G"))
    parser.add_argument("--time", default=os.getenv("REVISE_CACHE_SLURM_TIME", "12:00:00"))
    parser.add_argument("--account", default=os.getenv("REVISE_SLURM_ACCOUNT", "p32027"))
    parser.add_argument(
        "--conda-sh",
        default=os.getenv(
            "REVISE_CONDA_SH",
            "/gpfs/projects/p32027/conda/miniconda3/etc/profile.d/conda.sh",
        ),
    )
    parser.add_argument("--conda-env", default=os.getenv("REVISE_CONDA_ENV", "verlrun"))
    parser.add_argument("--yt-dlp-timeout-s", type=int, default=900)
    args = parser.parse_args()

    datasets = args.dataset or ["lvbench"]
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive.")

    run_name = args.run_name or "cache-videos-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root).resolve() / run_name
    scripts_dir = run_dir / "jobs"
    logs_dir = run_dir / "slurm"
    results_dir = run_dir / "results"
    for path in (scripts_dir, logs_dir, results_dir):
        path.mkdir(parents=True, exist_ok=True)

    video_cache_dir = Path(args.video_cache_dir).expanduser().resolve()
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "video_cache_dir": str(video_cache_dir),
        "submit": bool(args.submit),
        "slurm": {
            "partition": args.partition,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time": args.time,
            "account": args.account,
            "conda_sh": args.conda_sh,
            "conda_env": args.conda_env,
        },
        "chunks": [],
    }

    for dataset in datasets:
        split = args.split or ("test" if dataset == "videomme" else "train")
        total = _distinct_video_count(dataset, split)
        for start_idx in range(0, total, args.chunk_size):
            max_videos = min(args.chunk_size, total - start_idx)
            chunk_id = f"{dataset}-{split}-{start_idx:04d}-{start_idx + max_videos - 1:04d}"
            chunk_manifest = results_dir / f"{chunk_id}.json"
            job_path = scripts_dir / f"{chunk_id}.sbatch"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "repro" / "cache_hf_video_benchmark.py"),
                "--dataset",
                dataset,
                "--split",
                split,
                "--video-cache-dir",
                str(video_cache_dir),
                "--start-idx",
                str(start_idx),
                "--max-videos",
                str(max_videos),
                "--yt-dlp-timeout-s",
                str(args.yt_dlp_timeout_s),
                "--manifest",
                str(chunk_manifest),
            ]
            job_path.write_text(
                _sbatch_script(
                    dataset=dataset,
                    split=split,
                    start_idx=start_idx,
                    max_videos=max_videos,
                    manifest_path=chunk_manifest,
                    video_cache_dir=video_cache_dir,
                    conda_sh=args.conda_sh,
                    conda_env=args.conda_env,
                    partition=args.partition,
                    cpus_per_task=args.cpus_per_task,
                    mem=args.mem,
                    time_limit=args.time,
                    account=args.account,
                    yt_dlp_timeout_s=args.yt_dlp_timeout_s,
                    logs_dir=logs_dir,
                ),
                encoding="utf-8",
            )
            entry: dict[str, Any] = {
                "id": chunk_id,
                "dataset": dataset,
                "split": split,
                "start_idx": start_idx,
                "max_videos": max_videos,
                "total_distinct_videos": total,
                "job_script": str(job_path),
                "result_manifest": str(chunk_manifest),
                "command": shlex.join(command),
            }
            if args.submit:
                proc = subprocess.run(["sbatch", str(job_path)], capture_output=True, text=True, check=False)
                entry["sbatch_returncode"] = proc.returncode
                entry["sbatch_stdout"] = (proc.stdout or "").strip()
                entry["sbatch_stderr"] = (proc.stderr or "").strip()
                if proc.returncode == 0:
                    parts = entry["sbatch_stdout"].split()
                    entry["job_id"] = parts[-1] if parts else None
            manifest["chunks"].append(entry)

    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"run_dir": str(run_dir), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
