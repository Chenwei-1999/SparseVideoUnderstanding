#!/usr/bin/env python3
"""Phase A: submit Qwen2.5-VL-72B teacher data generation as Slurm jobs.

The 72B teacher replaces the GPT-4o teacher used in the published paper for
the SFT distillation step. This emits two sbatch jobs (one per dataset):

  - nextqa_teacher72b   :: generates SFT JSONL from NExT-QA train CSV
  - videoespresso_teacher72b :: generates SFT JSONL from VideoEspresso train JSON

Each job:
  * runs `examples/revise/run_generate_teacher_data{,_videoespresso}.sh`
  * with TEACHER_MODEL_PATH=$REVISE_QWEN25_VL_72B_PATH, MAX_SAMPLES=8000,
    TENSOR_PARALLEL_SIZE=2, GPU_MEMORY_UTILIZATION=0.85
  * on 2x A100-80GB (gres=gpu:a100:2), gengpu partition, 48h wall time
  * writes teacher JSONL + server log + summary into the run directory

After both jobs complete, the JSONL traces are the input to Phase E SFT/GRPO.

Usage:
  python scripts/repro/submit_phase_a_teacher72b_slurm.py \
      --output-root outputs/repro_runs \
      --run-name phase-a-teacher72b-20260523 --submit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Default asset paths come from $REVISE_ASSET_ROOT but we resolve the model
# path here so the sbatch script line is self-contained and stable across
# env changes between login submission and compute-node execution.
ASSET_ROOT = Path(os.environ.get("REVISE_ASSET_ROOT", str(Path.cwd() / "data" / "revise_assets"))).resolve()

# Two supported teacher variants. AWQ-INT4 is the default because the BF16
# 72B at TP=4 on 4xA100-80GB has 13+h of queue wait under typical load
# (only 9 four-A100 nodes cluster-wide). AWQ-INT4 fits at TP=2 on the much
# larger 2xA100 pool (~13 nodes) with ~18GB of weights/GPU and runs
# 3-5x faster per call. The bf16 variant is retained for parity reruns.
TEACHER_VARIANTS: dict[str, dict[str, object]] = {
    "awq": {
        "local_name": "Qwen2.5-VL-72B-Instruct-AWQ",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.85,
        "gres": "gpu:a100:2",
    },
    "bf16": {
        "local_name": "Qwen2.5-VL-72B-Instruct",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.85,
        "gres": "gpu:a100:4",
    },
}

# 8k samples per dataset matches the original paper SFT budget; the design
# spec caps wall time at 18h (we request 48h on `gengpu` for headroom).
MAX_SAMPLES = 8000
SERVER_TIMEOUT_S = 3600  # 72B startup is slower than 7B
# A successful Phase A teacher run must answer at least this fraction of the
# requested samples; otherwise the SFT input would be too sparse to be useful.
MIN_VALID_SAMPLE_FRACTION = 0.5

DATASETS: dict[str, dict[str, str]] = {
    "nextqa_teacher72b": {
        "script": "examples/revise/run_generate_teacher_data.sh",
        "env": (
            "VIDEO_ROOT={asset}/NExT-QA/NExTVideo "
            "MAP_JSON={asset}/NExT-QA/map_vid_vidorID.json "
            "CSV={asset}/NExT-QA/nextqa/train.csv "
        ),
        "log_basename": "nextqa_teacher72b_train_log.jsonl",
        "summary_basename": "nextqa_teacher72b_train_summary.json",
        "server_log_basename": "nextqa_teacher72b_train.server.log",
    },
    "videoespresso_teacher72b": {
        "script": "examples/revise/run_generate_teacher_data_videoespresso.sh",
        # The VideoEspresso teacher script invokes plug_and_play_egoschema_vllm.py,
        # whose loader requires `options/choices`+`correct_answer`. The raw
        # train JSON is open-ended; we feed the prepared MC variant produced
        # by prepare_videoespresso_mc_train.py instead.
        "env": (
            "VIDEO_ROOT={asset}/VideoEspresso "
            "JSON={asset}/VideoEspresso/train_video/videoespresso_train_mc.json "
        ),
        "log_basename": "videoespresso_teacher72b_train_log.jsonl",
        "summary_basename": "videoespresso_teacher72b_train_summary.json",
        "server_log_basename": "videoespresso_teacher72b_train.server.log",
    },
}


def _sbatch_script(
    *,
    name: str,
    spec: dict[str, str],
    variant: dict[str, object],
    asset_root: Path,
    teacher_path: Path,
    run_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    account: str,
    conda_sh: Path,
    conda_env: str,
) -> str:
    results_dir = run_dir / "results"
    log_path = results_dir / spec["log_basename"]
    summary_path = results_dir / spec["summary_basename"]
    server_log = results_dir / spec["server_log_basename"]
    slurm_log = run_dir / "slurm"
    slurm_log.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    hf_home = asset_root / "hf_home"
    extra_env = spec["env"].format(asset=asset_root)

    min_valid = int(MAX_SAMPLES * MIN_VALID_SAMPLE_FRACTION)
    tp_size = int(variant["tensor_parallel_size"])
    gpu_util = float(variant["gpu_memory_utilization"])
    gres = str(variant["gres"])
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=revise-{name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={slurm_log}/%x-%j.out
#SBATCH --error={slurm_log}/%x-%j.err

set -euo pipefail

cd {REPO_ROOT}
source {conda_sh}
conda activate {conda_env}
module load git 2>/dev/null || true

# Slurm inherits the submitter's env by default. Any pre-existing
# TEACHER_BASE_URL / BASE_URL / TEACHER_MODEL_ID / MODEL_ID / OpenAI vars
# would cause the run-script to route through an API endpoint instead of
# the local 72B vLLM server we explicitly request below.
unset TEACHER_BASE_URL BASE_URL TEACHER_MODEL_ID MODEL_ID \\
      OPENAI_BASE_URL OPENAI_API_KEY OPENAI_API_BASE 2>/dev/null || true

export REVISE_ASSET_ROOT={asset_root}
export HF_HOME={hf_home}
export HF_HUB_CACHE={hf_home}/hub
export HF_DATASETS_CACHE={hf_home}/datasets
export HF_XET_CACHE={hf_home}/xet
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Preflight: doctor snapshot at job start so we can debug env drift later.
python scripts/repro/doctor.py > {run_dir}/logs.doctor.{name}.$SLURM_JOB_ID.txt 2>&1 || true

# Run the existing teacher-generation shell with the 72B teacher. SUMMARY_JSON
# is not respected by the helpers; pass --summary-json through as extra args
# so the runner records failed/sample counts.
PYTHON_BIN=/projects/p32027/conda/miniconda3/envs/verlrun/bin/python \\
TEACHER_MODEL_PATH={teacher_path} \\
MAX_SAMPLES={MAX_SAMPLES} \\
TENSOR_PARALLEL_SIZE={tp_size} \\
GPU_MEMORY_UTILIZATION={gpu_util} \\
SERVER_TIMEOUT_S={SERVER_TIMEOUT_S} \\
LOG_PATH={log_path} \\
SERVER_LOG={server_log} \\
{extra_env}\\
./{spec["script"]} --summary-json {summary_path}

# Post-run validation: require at least {min_valid} distinct sample_id rows
# in the teacher JSONL before declaring the job successful. A run that exits
# zero but produced too few samples must not be accepted as Phase E input.
python - <<PYEOF
import json, sys
from pathlib import Path
log = Path("{log_path}")
if not log.exists():
    print(f"FAIL: teacher log {{log}} does not exist", file=sys.stderr)
    sys.exit(3)
ids = set()
for line in log.open():
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        continue
    sid = rec.get("sample_id") or rec.get("id") or rec.get("video_id")
    if sid is not None:
        ids.add(sid)
n = len(ids)
print(f"teacher_log_unique_samples={{n}} required>={min_valid}")
if n < {min_valid}:
    print(f"FAIL: only {{n}} distinct samples in teacher log; require >= {min_valid}", file=sys.stderr)
    sys.exit(4)
print("OK: teacher log meets minimum sample threshold")
PYEOF
"""


def _submit(scripts: dict[str, Path], *, submit: bool) -> tuple[dict[str, str], list[str]]:
    job_ids: dict[str, str] = {}
    failed: list[str] = []
    for name, script in scripts.items():
        if not submit:
            job_ids[name] = "(not submitted; use --submit)"
            continue
        proc = subprocess.run(
            ["sbatch", str(script)], capture_output=True, text=True, check=False
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode == 0:
            job_ids[name] = proc.stdout.strip().split()[-1]
        else:
            job_ids[name] = f"ERR rc={proc.returncode}: {(proc.stderr or proc.stdout).strip()[:200]}"
            failed.append(name)
    return job_ids, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Submit Phase A 72B teacher generation Slurm jobs.")
    ap.add_argument("--output-root", default="outputs/repro_runs")
    ap.add_argument("--run-name", default=time.strftime("phase-a-teacher72b-%Y%m%d-%H%M%S"))
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), action="append")
    ap.add_argument("--partition", default="gengpu")
    ap.add_argument("--cpus-per-task", type=int, default=16)
    ap.add_argument("--mem", default="128G")
    ap.add_argument("--time", default="2-00:00:00")
    ap.add_argument("--account", default="p32027")
    ap.add_argument(
        "--conda-sh", default="/gpfs/projects/p32027/conda/miniconda3/etc/profile.d/conda.sh"
    )
    ap.add_argument("--conda-env", default="verlrun")
    ap.add_argument("--asset-root", default=str(ASSET_ROOT))
    ap.add_argument(
        "--teacher-variant",
        default="awq",
        choices=list(TEACHER_VARIANTS.keys()),
        help="awq = INT4 AWQ-72B (TP=2 on 2xA100); bf16 = full-precision (TP=4 on 4xA100).",
    )
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args(argv)

    asset_root = Path(args.asset_root).expanduser().resolve()
    variant = TEACHER_VARIANTS[args.teacher_variant]
    teacher_path = asset_root / "models" / variant["local_name"]
    if not teacher_path.exists() or not any(teacher_path.glob("*.safetensors")):
        print(f"ERROR: {args.teacher_variant} teacher not found at {teacher_path}", file=sys.stderr)
        return 2

    run_dir = (REPO_ROOT / args.output_root / args.run_name).resolve()
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    selected = args.dataset or list(DATASETS.keys())
    scripts: dict[str, Path] = {}
    for name in selected:
        spec = DATASETS[name]
        script = jobs_dir / f"{name}.sbatch"
        script.write_text(
            _sbatch_script(
                name=name,
                spec=spec,
                variant=variant,
                asset_root=asset_root,
                teacher_path=teacher_path,
                run_dir=run_dir,
                partition=args.partition,
                cpus=args.cpus_per_task,
                mem=args.mem,
                time_limit=args.time,
                account=args.account,
                conda_sh=Path(args.conda_sh),
                conda_env=args.conda_env,
            )
        )
        script.chmod(0o755)
        scripts[name] = script

    job_ids, failed = _submit(scripts, submit=args.submit)
    manifest = {
        "phase": "A",
        "asset_root": str(asset_root),
        "teacher_variant": args.teacher_variant,
        "teacher_model_path": str(teacher_path),
        "max_samples": MAX_SAMPLES,
        "tensor_parallel_size": int(variant["tensor_parallel_size"]),
        "gres": str(variant["gres"]),
        "run_dir": str(run_dir),
        "submitted": bool(args.submit),
        "scripts": {k: str(v) for k, v in scripts.items()},
        "job_ids": job_ids,
        "failed_to_submit": failed,
        "partition": args.partition,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if failed:
        print(f"sbatch submission failed for: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
