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

from scripts.repro.common import discover_assets
from scripts.repro.paper_suite import EXPERIMENTS, cmd_to_text


DEFAULT_SLURM_EXPERIMENTS = [
    # NExT-QA family
    "nextqa_pnp",
    "nextqa_oneshot",
    "nextqa_caption",
    "nextqa_videoagent",
    "nextqa_videoagent_official",
    # VideoEspresso main + ablations
    "videoespresso_pnp",
    "videoespresso_oneshot",
    "videoespresso_budget_10x2",
    "videoespresso_budget_2x10",
    "videoespresso_budget_4x5",
    "videoespresso_budget_5x4",
    "videoespresso_budget_7x4",
    "videoespresso_budget_9x4",
    "videoespresso_best_01x06",
    "videoespresso_best_02x04",
    "videoespresso_best_03x06",
    "videoespresso_best_04x04",
    "videoespresso_components_full",
    "videoespresso_components_no_carryover",
    "videoespresso_components_no_structured",
    "videoespresso_components_no_both",
    # EgoSchema
    "egoschema_pnp",
    # Video-MME + LVBench (both Qwen2.5-VL-7B and exact LLaVA-OV-7B backbones)
    "videomme_pnp",
    "videomme_pnp_hf",
    "videomme_oneshot",
    "lvbench_pnp",
    "lvbench_pnp_hf",
    "lvbench_oneshot",
]


def _run_text(cmd: list[str], *, cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return (proc.stdout or proc.stderr or "").strip()


def _git_snapshot() -> dict[str, Any]:
    return {
        "head": _run_text(["git", "rev-parse", "HEAD"]),
        "branch": _run_text(["git", "branch", "--show-current"]),
        "status_short": _run_text(["git", "status", "--short"]),
        "diff_cached_stat": _run_text(["git", "diff", "--cached", "--stat"]),
        "diff_stat": _run_text(["git", "diff", "--stat"]),
    }


def _selected_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(DEFAULT_SLURM_EXPERIMENTS)
    if not args.experiment:
        raise SystemExit("Specify --experiment or --all.")
    unknown = [eid for eid in args.experiment if eid not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment ids: {', '.join(unknown)}")
    return list(args.experiment)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sbatch_script(
    *,
    exp_id: str,
    command: str,
    run_dir: Path,
    conda_sh: str,
    conda_env: str,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    account: str,
) -> str:
    logs_dir = run_dir / "slurm"
    logs_dir.mkdir(parents=True, exist_ok=True)
    doctor_log = run_dir / "logs" / f"doctor.{exp_id}.$SLURM_JOB_ID.txt"
    account_line = f"#SBATCH --account={account}\n" if account else ""
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=revise-{exp_id}
#SBATCH --partition={partition}
{account_line}{gres_line}#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={logs_dir}/%x-%j.out
#SBATCH --error={logs_dir}/%x-%j.err

set -euo pipefail

cd {shlex.quote(str(REPO_ROOT))}
source {shlex.quote(conda_sh)}
conda activate {shlex.quote(conda_env)}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export REVISE_RUN_DIR={shlex.quote(str(run_dir))}
export HF_HOME="${{REVISE_HF_HOME:-{shlex.quote(str(run_dir / "hf_home"))}}}"
export HF_HUB_CACHE="${{HF_HUB_CACHE:-$HF_HOME/hub}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-$HF_HOME/datasets}}"
export HF_XET_CACHE="${{HF_XET_CACHE:-$HF_HOME/xet}}"
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-{shlex.quote(str(run_dir / "xdg_cache"))}}}"

python scripts/repro/doctor.py > "{doctor_log}" 2>&1 || true

{command}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and optionally submit Slurm jobs for paper reproduction.")
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--submit", action="store_true", help="Submit generated jobs with sbatch.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / "repro_runs"),
        help="Root directory for manifests, Slurm scripts, summaries, and JSONL prompt logs.",
    )
    parser.add_argument("--run-name", default="", help="Stable run directory name. Defaults to UTC timestamp.")
    parser.add_argument("--partition", default=os.getenv("REVISE_SLURM_PARTITION", "gengpu"))
    parser.add_argument("--gres", default=os.getenv("REVISE_SLURM_GRES", "gpu:a100:1"))
    parser.add_argument("--cpus-per-task", type=int, default=int(os.getenv("REVISE_SLURM_CPUS", "8")))
    parser.add_argument("--mem", default=os.getenv("REVISE_SLURM_MEM", "64G"))
    parser.add_argument("--time", default=os.getenv("REVISE_SLURM_TIME", "02:00:00"))
    parser.add_argument("--account", default=os.getenv("REVISE_SLURM_ACCOUNT", "p32027"))
    parser.add_argument(
        "--conda-sh",
        default=os.getenv(
            "REVISE_CONDA_SH",
            "/gpfs/projects/p32027/conda/miniconda3/etc/profile.d/conda.sh",
        ),
    )
    parser.add_argument("--conda-env", default=os.getenv("REVISE_CONDA_ENV", "verlrun"))
    args = parser.parse_args()

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root).resolve() / run_name
    results_dir = run_dir / "results"
    scripts_dir = run_dir / "jobs"
    logs_dir = run_dir / "logs"
    for path in (results_dir, scripts_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    assets = discover_assets()
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "smoke": bool(args.smoke),
        "submit": bool(args.submit),
        "slurm": {
            "partition": args.partition,
            "gres": args.gres,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time": args.time,
            "account": args.account,
            "conda_sh": args.conda_sh,
            "conda_env": args.conda_env,
        },
        "git": _git_snapshot(),
        "assets": assets,
        "experiments": [],
    }

    for exp_id in _selected_ids(args):
        meta = EXPERIMENTS[exp_id]
        check = meta["check"]  # type: ignore[index]
        build = meta["build"]  # type: ignore[index]
        missing = check(assets, bool(args.smoke))
        entry: dict[str, Any] = {
            "id": exp_id,
            "title": meta["title"],
            "paper_ref": meta["paper_ref"],
            "run_supported": bool(meta["run_supported"]),
            "blocked": bool(missing),
            "blocked_reasons": missing,
        }
        if missing:
            manifest["experiments"].append(entry)
            continue

        cmd = build(assets, bool(args.smoke), results_dir)
        command = cmd_to_text(cmd)
        entry["command"] = command
        job_path = scripts_dir / f"{exp_id}.sbatch"
        job_path.write_text(
            _sbatch_script(
                exp_id=exp_id,
                command=command,
                run_dir=run_dir,
                conda_sh=args.conda_sh,
                conda_env=args.conda_env,
                partition=args.partition,
                gres=args.gres,
                cpus_per_task=args.cpus_per_task,
                mem=args.mem,
                time_limit=args.time,
                account=args.account,
            ),
            encoding="utf-8",
        )
        entry["job_script"] = str(job_path)

        if args.submit:
            proc = subprocess.run(["sbatch", str(job_path)], capture_output=True, text=True, check=False)
            entry["sbatch_returncode"] = proc.returncode
            entry["sbatch_stdout"] = (proc.stdout or "").strip()
            entry["sbatch_stderr"] = (proc.stderr or "").strip()
            if proc.returncode == 0:
                parts = entry["sbatch_stdout"].split()
                entry["job_id"] = parts[-1] if parts else None
        manifest["experiments"].append(entry)

    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"run_dir": str(run_dir), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
