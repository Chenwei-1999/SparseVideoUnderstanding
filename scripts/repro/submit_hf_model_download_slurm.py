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

from download_hf_models import MODEL_SPECS, PAPER_MODEL_KEYS


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _selected_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []
    if args.all:
        keys.extend(MODEL_SPECS)
    if args.paper_models:
        keys.extend(PAPER_MODEL_KEYS)
    keys.extend(args.model or [])
    if not keys:
        raise SystemExit("Specify --model, --paper-models, or --all.")
    unknown = sorted({key for key in keys if key not in MODEL_SPECS})
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")
    return list(dict.fromkeys(keys))


def _sbatch_script(
    *,
    run_dir: Path,
    asset_root: Path,
    model_keys: list[str],
    conda_sh: str,
    conda_env: str,
    partition: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    account: str,
    concurrency: int,
) -> str:
    logs_dir = run_dir / "slurm"
    results_dir = run_dir / "results"
    account_line = f"#SBATCH --account={account}\n" if account else ""
    keys = " ".join(shlex.quote(key) for key in model_keys)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=revise-hf-models
#SBATCH --partition={partition}
{account_line}#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --array=1-{len(model_keys)}%{concurrency}
#SBATCH --output={logs_dir}/%x-%A_%a.out
#SBATCH --error={logs_dir}/%x-%A_%a.err

set -euo pipefail

cd {shlex.quote(str(REPO_ROOT))}
source {shlex.quote(conda_sh)}
conda activate {shlex.quote(conda_env)}

export PYTHONUNBUFFERED=1
export REVISE_ASSET_ROOT={shlex.quote(str(asset_root))}
export HF_HOME="${{REVISE_HF_HOME:-{shlex.quote(str(asset_root / ".hf_home"))}}}"
export HF_HUB_CACHE="${{HF_HUB_CACHE:-$HF_HOME/hub}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-$HF_HOME/datasets}}"
export HF_XET_CACHE="${{HF_XET_CACHE:-$HF_HOME/xet}}"

MODEL_KEYS=({keys})
MODEL_KEY="${{MODEL_KEYS[$((SLURM_ARRAY_TASK_ID - 1))]}}"

python scripts/repro/download_hf_models.py \\
  --model "${{MODEL_KEY}}" \\
  --asset-root {shlex.quote(str(asset_root))} \\
  --manifest "{results_dir}/model-${{MODEL_KEY}}.json"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit non-GPU Slurm jobs to download exact HF model snapshots.")
    parser.add_argument("--model", action="append", choices=sorted(MODEL_SPECS))
    parser.add_argument("--paper-models", action="store_true", help="Download Qwen2-VL-7B, InternVL2-8B, and LLaVA-OV-7B.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "repro_runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--asset-root",
        default=os.getenv("REVISE_ASSET_ROOT", str(REPO_ROOT / "data" / "revise_assets")),
    )
    parser.add_argument("--partition", default=os.getenv("REVISE_CACHE_SLURM_PARTITION", "normal"))
    parser.add_argument("--cpus-per-task", type=int, default=int(os.getenv("REVISE_CACHE_SLURM_CPUS", "4")))
    parser.add_argument("--mem", default=os.getenv("REVISE_MODEL_SLURM_MEM", "32G"))
    parser.add_argument("--time", default=os.getenv("REVISE_MODEL_SLURM_TIME", "08:00:00"))
    parser.add_argument("--account", default=os.getenv("REVISE_SLURM_ACCOUNT", "p32027"))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--conda-sh",
        default=os.getenv(
            "REVISE_CONDA_SH",
            "/gpfs/projects/p32027/conda/miniconda3/etc/profile.d/conda.sh",
        ),
    )
    parser.add_argument("--conda-env", default=os.getenv("REVISE_CONDA_ENV", "verlrun"))
    args = parser.parse_args()

    model_keys = _selected_keys(args)
    if args.concurrency <= 0 or args.concurrency > len(model_keys):
        raise SystemExit(f"--concurrency must be in 1..{len(model_keys)}.")

    run_name = args.run_name or "hf-models-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root).resolve() / run_name
    for path in (run_dir / "jobs", run_dir / "slurm", run_dir / "results"):
        path.mkdir(parents=True, exist_ok=True)
    asset_root = Path(args.asset_root).expanduser().resolve()
    job_path = run_dir / "jobs" / "hf-model-download.sbatch"
    job_path.write_text(
        _sbatch_script(
            run_dir=run_dir,
            asset_root=asset_root,
            model_keys=model_keys,
            conda_sh=args.conda_sh,
            conda_env=args.conda_env,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time_limit=args.time,
            account=args.account,
            concurrency=args.concurrency,
        ),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "job_script": str(job_path),
        "asset_root": str(asset_root),
        "submit": bool(args.submit),
        "models": [
            {
                "key": key,
                "repo_id": MODEL_SPECS[key].repo_id,
                "local_name": MODEL_SPECS[key].local_name,
                "env_var": MODEL_SPECS[key].env_var,
                "paper_rows": list(MODEL_SPECS[key].paper_rows),
            }
            for key in model_keys
        ],
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
