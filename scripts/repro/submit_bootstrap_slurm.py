#!/usr/bin/env python3
"""Submit Slurm jobs for the heavy bootstrap_assets.py steps.

The cheap steps (layout, symlinked models, symlinked VideoEspresso, egoschema
subset, LLaVA-NeXT) run inline on the login node in seconds. The heavy steps
(nextqa_videos ~23G, videomme ~100G, lvbench ~25G) are non-GPU but
network/IO-bound, so they're better off on the `normal` partition where they
won't time out a login session.

Each Slurm job re-invokes bootstrap_assets.py for that single step against the
target asset root, so retry/idempotency is delegated to bootstrap_assets.py.

Usage:
  python scripts/repro/submit_bootstrap_slurm.py \\
      --asset-root /path/to/your/scratch/revise_paper_repro \\
      --output-root outputs/repro_runs/bootstrap-20260523 \\
      --steps nextqa_videos videomme lvbench --submit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

HEAVY_STEPS = ("nextqa_videos", "videomme", "lvbench")

# Wall-time budget per step (conservative).
WALLTIME = {
    "nextqa_videos": "06:00:00",
    "videomme": "12:00:00",
    "lvbench": "06:00:00",
}


def _sbatch_script(
    *,
    step: str,
    asset_root: Path,
    run_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    account: str,
    conda_sh: Path,
    conda_env: str,
) -> str:
    log_root = run_dir / "slurm"
    log_root.mkdir(parents=True, exist_ok=True)
    hf_home = asset_root / "hf_home"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=bootstrap-{step}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={log_root}/%x-%j.out
#SBATCH --error={log_root}/%x-%j.err

set -euo pipefail

cd {REPO_ROOT}
source {conda_sh}
conda activate {conda_env}
# Compute nodes on Quest don't have git in PATH by default; needed for the
# NExT-QA annotations clone. `|| true` so absence of the module isn't fatal
# if git is already in PATH (annotations clone is idempotent anyway).
module load git 2>/dev/null || true

export REVISE_ASSET_ROOT={asset_root}
export HF_HOME={hf_home}
export HF_HUB_CACHE={hf_home}/hub
export HF_DATASETS_CACHE={hf_home}/datasets
export HF_XET_CACHE={hf_home}/xet
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
export PYTHONUNBUFFERED=1

python scripts/repro/bootstrap_assets.py {step} --asset-root {asset_root}
"""


def _submit(scripts: dict[str, Path], *, submit: bool) -> tuple[dict[str, str], list[str]]:
    """Submit each sbatch script. Returns (job_ids, failed_steps).

    `failed_steps` is non-empty if any sbatch call returned non-zero. Callers
    use it to set a non-zero process exit code so wrappers don't think a
    failed submission is success.
    """
    job_ids: dict[str, str] = {}
    failed: list[str] = []
    for step, script in scripts.items():
        if not submit:
            job_ids[step] = "(not submitted; use --submit)"
            continue
        proc = subprocess.run(
            ["sbatch", str(script)], capture_output=True, text=True, check=False
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode == 0:
            # sbatch prints "Submitted batch job 12345"
            job_ids[step] = proc.stdout.strip().split()[-1]
        else:
            job_ids[step] = f"ERR rc={proc.returncode}: {(proc.stderr or proc.stdout).strip()[:200]}"
            failed.append(step)
    return job_ids, failed


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Submit bootstrap_assets heavy steps via Slurm.")
    ap.add_argument(
        "--asset-root",
        default=os.environ.get("REVISE_ASSET_ROOT", str(Path.cwd() / "data" / "revise_assets")),
    )
    ap.add_argument(
        "--output-root",
        default="outputs/repro_runs",
    )
    ap.add_argument(
        "--run-name",
        default=time.strftime("bootstrap-%Y%m%d-%H%M%S"),
        help="Subdirectory under --output-root for manifest + sbatch + logs.",
    )
    ap.add_argument(
        "--steps",
        nargs="+",
        default=list(HEAVY_STEPS),
        choices=HEAVY_STEPS,
    )
    ap.add_argument("--partition", default="normal")
    ap.add_argument("--cpus-per-task", type=int, default=8)
    ap.add_argument("--mem", default="32G")
    ap.add_argument("--account", default="p32027")
    ap.add_argument(
        "--conda-sh", default="/gpfs/projects/p32027/conda/miniconda3/etc/profile.d/conda.sh"
    )
    ap.add_argument("--conda-env", default="verlrun")
    ap.add_argument("--submit", action="store_true", help="Submit generated jobs with sbatch.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    asset_root = Path(args.asset_root).expanduser().resolve()
    run_dir = (REPO_ROOT / args.output_root / args.run_name).resolve()
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    scripts: dict[str, Path] = {}
    for step in args.steps:
        script = jobs_dir / f"{step}.sbatch"
        script.write_text(
            _sbatch_script(
                step=step,
                asset_root=asset_root,
                run_dir=run_dir,
                partition=args.partition,
                cpus=args.cpus_per_task,
                mem=args.mem,
                time_limit=WALLTIME.get(step, "08:00:00"),
                account=args.account,
                conda_sh=Path(args.conda_sh),
                conda_env=args.conda_env,
            )
        )
        script.chmod(0o755)
        scripts[step] = script

    job_ids, failed_steps = _submit(scripts, submit=args.submit)

    manifest = {
        "asset_root": str(asset_root),
        "run_dir": str(run_dir),
        "submitted": bool(args.submit),
        "scripts": {k: str(v) for k, v in scripts.items()},
        "job_ids": job_ids,
        "failed_to_submit": failed_steps,
        "partition": args.partition,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if failed_steps:
        # Non-zero exit so wrappers / scripts don't treat a failed submission as success.
        print(f"sbatch submission failed for: {failed_steps}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
