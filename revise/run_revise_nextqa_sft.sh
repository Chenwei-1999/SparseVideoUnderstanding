#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./revise/run_revise_nextqa_sft.sh [hydra overrides ...]
#
# Generates SFT data from configured teacher TRAIN-split logs (if not already present),
# then runs FSDP SFT training on Qwen2.5-VL-3B to teach the REVISE format.
#
# Prerequisites:
#   1. Generate teacher data on train split first:
#      ./revise/run_generate_teacher_data.sh

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/revise/config"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
ASSET_ROOT="${REVISE_ASSET_ROOT:-$PROJECT_DIR/data/revise_assets}"
SFT_CONFIG_NAME="${SFT_CONFIG_NAME:-revise_nextqa_sft}"

SFT_INPUT="${SFT_INPUT:-$PROJECT_DIR/outputs/nextqa_teacher_train_log.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-$PROJECT_DIR/outputs/sft_data/revise_sft.parquet}"
SFT_GENERATE_ARGS="${SFT_GENERATE_ARGS:-}"
N_GPUS="${N_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}"
BASE_MODEL="${BASE_MODEL:-${REVISE_QWEN25_VL_3B_PATH:-$ASSET_ROOT/models/Qwen2.5-VL-3B-Instruct}}"
HYDRA_ARGS=("$@")
HAS_PARTIAL_PRETRAIN=0
for arg in "${HYDRA_ARGS[@]}"; do
    if [[ "$arg" == model.partial_pretrain=* ]]; then
        HAS_PARTIAL_PRETRAIN=1
        break
    fi
done
if [ "$HAS_PARTIAL_PRETRAIN" -eq 0 ]; then
    HYDRA_ARGS+=("model.partial_pretrain=$BASE_MODEL")
fi

# Step 1: Generate SFT parquet data if not present
TRAIN_FILE="${SFT_OUTPUT%.parquet}_train.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    if [ ! -f "$SFT_INPUT" ]; then
        echo "ERROR: Teacher data not found at $SFT_INPUT"
        echo "Run ./revise/run_generate_teacher_data.sh first to generate train-split teacher data."
        exit 1
    fi
    echo "=== Generating SFT data ==="
    # SFT_GENERATE_ARGS may include curation flags such as:
    #   --min-first-select-ratio 0.35
    # Keep this as a simple escape hatch so the default paper path remains unchanged.
    # shellcheck disable=SC2086
    "$PYTHON_BIN" "$PROJECT_DIR/revise/generate_sft_data.py" \
        --input "$SFT_INPUT" \
        --output "$SFT_OUTPUT" \
        $SFT_GENERATE_ARGS
else
    echo "=== SFT data already exists at $TRAIN_FILE, skipping generation ==="
fi

# Step 2: Run SFT training
echo "=== Starting SFT training ==="
echo "Torchrun master port: $MASTER_PORT"
"$TORCHRUN_BIN" --nproc_per_node="$N_GPUS" \
    --master_port="$MASTER_PORT" \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path "$CONFIG_PATH" \
    --config-name "$SFT_CONFIG_NAME" \
    "${HYDRA_ARGS[@]}"

# Step 3: Copy preprocessor_config.json to HF checkpoint (needed for vLLM VLM loading)
SFT_CKPT_DIR="${SFT_CKPT_DIR:-$PROJECT_DIR/outputs/revise_nextqa_sft}"
write_sft_provenance() {
    local out_path="$1"
    local checkpoint_path="$2"
    SFT_INPUT="$SFT_INPUT" \
    SFT_OUTPUT="$SFT_OUTPUT" \
    SFT_GENERATE_ARGS="$SFT_GENERATE_ARGS" \
    BASE_MODEL="$BASE_MODEL" \
    SFT_CONFIG_NAME="$SFT_CONFIG_NAME" \
    N_GPUS="$N_GPUS" \
    "$PYTHON_BIN" - "$out_path" "$checkpoint_path" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path


def flag_value(parts, name, default):
    for idx, part in enumerate(parts):
        if part == name and idx + 1 < len(parts):
            return parts[idx + 1]
    return default


out_path = Path(sys.argv[1])
checkpoint_path = sys.argv[2]
generate_args = os.environ.get("SFT_GENERATE_ARGS", "")
parts = shlex.split(generate_args)
max_rounds = int(flag_value(parts, "--max-rounds", "4"))
min_first_select_ratio = float(flag_value(parts, "--min-first-select-ratio", "0"))
manifest = {
    "setting": "nextqa_table4_sft",
    "dataset": "nextqa",
    "sft_path": checkpoint_path,
    "sft_input": os.environ.get("SFT_INPUT", ""),
    "sft_output": os.environ.get("SFT_OUTPUT", ""),
    "base_model": os.environ.get("BASE_MODEL", ""),
    "config_name": os.environ.get("SFT_CONFIG_NAME", ""),
    "n_gpus": int(os.environ.get("N_GPUS", "0") or 0),
    "sft_generate": {
        "raw_args": generate_args,
        "max_rounds": max_rounds,
        "min_first_select_ratio": min_first_select_ratio,
        "variable_length_traces": max_rounds > 1,
    },
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
if [ -d "$SFT_CKPT_DIR" ]; then
    write_sft_provenance "$SFT_CKPT_DIR/revise_sft_provenance.json" "$SFT_CKPT_DIR"
fi
for hf_dir in "$SFT_CKPT_DIR"/global_step_*/hf_model "$SFT_CKPT_DIR"/global_step_*/huggingface "$SFT_CKPT_DIR"/hf_model "$SFT_CKPT_DIR"/huggingface; do
    if [ -d "$hf_dir" ]; then
        if [ ! -f "$hf_dir/preprocessor_config.json" ]; then
            cp "$BASE_MODEL/preprocessor_config.json" "$hf_dir/"
            echo "Copied preprocessor_config.json to $hf_dir"
        fi
        write_sft_provenance "$hf_dir/revise_sft_provenance.json" "$hf_dir"
    fi
done

echo "=== SFT training complete ==="
