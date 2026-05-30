#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sbatch_script(
    *,
    run_dir: Path,
    video_cache_dir: Path,
    conda_sh: str,
    conda_env: str,
    partition: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    account: str,
    concurrency: int,
    delete_zip_after_extract: bool,
    overwrite_existing: bool,
) -> str:
    logs_dir = run_dir / "slurm"
    results_dir = run_dir / "results"
    account_line = f"#SBATCH --account={account}\n" if account else ""
    delete_flag = " --delete-zip-after-extract" if delete_zip_after_extract else ""
    overwrite_flag = " --overwrite-existing" if overwrite_existing else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=revise-lvbench-official
#SBATCH --partition={partition}
{account_line}#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --array=1-14%{concurrency}
#SBATCH --output={logs_dir}/%x-%A_%a.out
#SBATCH --error={logs_dir}/%x-%A_%a.err

set -euo pipefail

cd {shlex.quote(str(REPO_ROOT))}
source {shlex.quote(conda_sh)}
conda activate {shlex.quote(conda_env)}

export PYTHONUNBUFFERED=1
export HF_HOME="${{REVISE_HF_HOME:-{shlex.quote(str(REPO_ROOT / "data" / "revise_assets" / ".hf_home"))}}}"
export HF_HUB_CACHE="${{HF_HUB_CACHE:-$HF_HOME/hub}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-$HF_HOME/datasets}}"
export HF_XET_CACHE="${{HF_XET_CACHE:-$HF_HOME/xet}}"

python scripts/repro/download_lvbench_official_videos.py \\
  --chunk "${{SLURM_ARRAY_TASK_ID}}" \\
  --video-cache-dir {shlex.quote(str(video_cache_dir))} \\
  --manifest "{results_dir}/lvbench-official-chunk-${{SLURM_ARRAY_TASK_ID}}.json"{delete_flag}{overwrite_flag}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit official LVBench HF video chunk download/extract jobs.")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "repro_runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--video-cache-dir",
        default=os.getenv("REVISE_VIDEO_CACHE_DIR", str(REPO_ROOT / "data" / "revise_assets" / "video_cache")),
    )
    parser.add_argument("--partition", default=os.getenv("REVISE_CACHE_SLURM_PARTITION", "normal"))
    parser.add_argument("--cpus-per-task", type=int, default=int(os.getenv("REVISE_CACHE_SLURM_CPUS", "4")))
    parser.add_argument("--mem", default=os.getenv("REVISE_CACHE_SLURM_MEM", "24G"))
    parser.add_argument("--time", default=os.getenv("REVISE_CACHE_SLURM_TIME", "12:00:00"))
    parser.add_argument("--account", default=os.getenv("REVISE_SLURM_ACCOUNT", "p32027"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--keep-zips", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--conda-sh",
        default=os.getenv(
            "REVISE_CONDA_SH",
            "/gpfs/projects/p32027/conda/miniconda3/etc/profile.d/conda.sh",
        ),
    )
    parser.add_argument("--conda-env", default=os.getenv("REVISE_CONDA_ENV", "verlrun"))
    args = parser.parse_args()

    if args.concurrency <= 0 or args.concurrency > 14:
        raise SystemExit("--concurrency must be in 1..14.")

    run_name = args.run_name or "lvbench-official-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root).resolve() / run_name
    for path in (run_dir / "jobs", run_dir / "slurm", run_dir / "results"):
        path.mkdir(parents=True, exist_ok=True)
    video_cache_dir = Path(args.video_cache_dir).expanduser().resolve()
    job_path = run_dir / "jobs" / "lvbench-official-download.sbatch"
    job_path.write_text(
        _sbatch_script(
            run_dir=run_dir,
            video_cache_dir=video_cache_dir,
            conda_sh=args.conda_sh,
            conda_env=args.conda_env,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time_limit=args.time,
            account=args.account,
            concurrency=args.concurrency,
            delete_zip_after_extract=not args.keep_zips,
            overwrite_existing=bool(args.overwrite_existing),
        ),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "job_script": str(job_path),
        "video_cache_dir": str(video_cache_dir),
        "submit": bool(args.submit),
        "chunks": list(range(1, 15)),
        "slurm": {
            "partition": args.partition,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time": args.time,
            "account": args.account,
            "concurrency": args.concurrency,
            "conda_sh": args.conda_sh,
            "conda_env": args.conda_env,
        },
        "delete_zip_after_extract": not args.keep_zips,
        "overwrite_existing": bool(args.overwrite_existing),
    }
    if args.submit:
        proc = subprocess.run(["sbatch", str(job_path)], capture_output=True, text=True, check=False)
        manifest["sbatch_returncode"] = proc.returncode
        manifest["sbatch_stdout"] = (proc.stdout or "").strip()
        manifest["sbatch_stderr"] = (proc.stderr or "").strip()
        if proc.returncode == 0:
            parts = manifest["sbatch_stdout"].split()
            manifest["job_id"] = parts[-1] if parts else None
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"run_dir": str(run_dir), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
