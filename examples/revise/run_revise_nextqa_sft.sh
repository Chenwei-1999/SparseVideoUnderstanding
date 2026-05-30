#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./examples/revise/run_revise_nextqa_sft.sh [hydra overrides ...]
#
# Generates SFT data from configured teacher TRAIN-split logs (if not already present),
# then runs FSDP SFT training on Qwen2.5-VL-3B to teach the REVISE format.
#
# Prerequisites:
#   1. Generate teacher data on train split first:
#      ./examples/revise/run_generate_teacher_data.sh

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/revise/config"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
ASSET_ROOT="${REVISE_ASSET_ROOT:-$PROJECT_DIR/data/revise_assets}"
SFT_CONFIG_NAME="${SFT_CONFIG_NAME:-revise_nextqa_sft}"

SFT_INPUT="${SFT_INPUT:-$PROJECT_DIR/outputs/nextqa_teacher_train_log.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-$PROJECT_DIR/outputs/sft_data/revise_sft.parquet}"
N_GPUS="${N_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}"

# Step 1: Generate SFT parquet data if not present
TRAIN_FILE="${SFT_OUTPUT%.parquet}_train.parquet"
if [ ! -f "$TRAIN_FILE" ]; then
    if [ ! -f "$SFT_INPUT" ]; then
        echo "ERROR: Teacher data not found at $SFT_INPUT"
        echo "Run ./examples/revise/run_generate_teacher_data.sh first to generate train-split teacher data."
        exit 1
    fi
    echo "=== Generating SFT data ==="
    "$PYTHON_BIN" "$PROJECT_DIR/examples/revise/generate_sft_data.py" \
        --input "$SFT_INPUT" \
        --output "$SFT_OUTPUT"
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
    "$@"

# Step 3: Copy preprocessor_config.json to HF checkpoint (needed for vLLM VLM loading)
BASE_MODEL="${BASE_MODEL:-${REVISE_QWEN25_VL_3B_PATH:-$ASSET_ROOT/models/Qwen2.5-VL-3B-Instruct}}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-$PROJECT_DIR/outputs/revise_nextqa_sft}"
for hf_dir in "$SFT_CKPT_DIR"/global_step_*/hf_model "$SFT_CKPT_DIR"/global_step_*/huggingface "$SFT_CKPT_DIR"/hf_model "$SFT_CKPT_DIR"/huggingface; do
    if [ -d "$hf_dir" ] && [ ! -f "$hf_dir/preprocessor_config.json" ]; then
        cp "$BASE_MODEL/preprocessor_config.json" "$hf_dir/"
        echo "Copied preprocessor_config.json to $hf_dir"
    fi
done

echo "=== SFT training complete ==="
