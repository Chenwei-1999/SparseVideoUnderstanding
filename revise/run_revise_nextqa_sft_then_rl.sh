#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ENGINE=vllm ./revise/run_revise_nextqa_sft_then_rl.sh
#
# Full pipeline:
#   1. Generate teacher data (configured teacher on train split) — if not already done
#   2. Convert to SFT parquet
#   3. SFT training (format fine-tuning)
#   4. RL training (GRPO + EAGER)
#
# Prerequisites:
#   Teacher data must be generated first:
#     ./revise/run_generate_teacher_data.sh
#
# For the audited Table 4 RL-after-SFT row, prefer:
#   python scripts/paper_suite.py run --experiment nextqa_table4_rl_after_sft

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/revise/config"
ENGINE=${ENGINE:-vllm}
N_GPUS=${N_GPUS:-4}
PYTHON_BIN="${PYTHON_BIN:-python3}"

SFT_CKPT_DIR="${SFT_CKPT_DIR:-$PROJECT_DIR/outputs/revise_nextqa_sft}"
TEACHER_LOG="${TEACHER_LOG:-$PROJECT_DIR/outputs/nextqa_teacher_train_log.jsonl}"
read -r -a SFT_EXTRA_ARGS_ARRAY <<< "${SFT_EXTRA_ARGS:-}"
read -r -a RL_EXTRA_ARGS_ARRAY <<< "${RL_EXTRA_ARGS:-}"

# ── Check prerequisites ───────────────────────────────────────────────────────
if [ ! -f "$TEACHER_LOG" ]; then
    echo "ERROR: Teacher data log not found at $TEACHER_LOG"
    echo ""
    echo "Run the teacher data generation first:"
    echo "  ./revise/run_generate_teacher_data.sh"
    echo ""
    echo "This runs the configured teacher on the NExT-QA TRAIN split"
    echo "(not val split, to avoid data leakage)."
    exit 1
fi

# ── Stage 1: SFT ──────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Stage 1: SFT (format fine-tuning)"
echo "================================================================"
SFT_INPUT="$TEACHER_LOG" ./revise/run_revise_nextqa_sft.sh "${SFT_EXTRA_ARGS_ARRAY[@]}"

# ── Find latest HF checkpoint ─────────────────────────────────────────────────
LATEST_STEP=$(ls -d "$SFT_CKPT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -z "$LATEST_STEP" ]; then
    echo "ERROR: No SFT checkpoint found in $SFT_CKPT_DIR"
    exit 1
fi

HF_MODEL_PATH=""
for candidate in "$LATEST_STEP/hf_model" "$LATEST_STEP/huggingface" "$SFT_CKPT_DIR/hf_model" "$SFT_CKPT_DIR/huggingface"; do
    if [ -d "$candidate" ]; then
        HF_MODEL_PATH="$candidate"
        break
    fi
done

if [ -z "$HF_MODEL_PATH" ]; then
    echo "ERROR: HF model not found at $HF_MODEL_PATH"
    echo "Available dirs under latest step:"
    ls "$LATEST_STEP"
    exit 1
fi

echo ""
echo "  SFT checkpoint: $HF_MODEL_PATH"
echo ""

# ── Stage 2: RL (GRPO + EAGER) ────────────────────────────────────────────────
echo "================================================================"
echo "  Stage 2: RL training (GRPO + EAGER reward)"
echo "================================================================"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path "$CONFIG_PATH" \
  --config-name revise_nextqa_grpo_after_sft \
  actor_rollout_ref.model.path="$HF_MODEL_PATH" \
  actor_rollout_ref.rollout.name="$ENGINE" \
  trainer.logger='["console","wandb"]' \
  "${RL_EXTRA_ARGS_ARRAY[@]}" \
  "$@"

echo ""
echo "================================================================"
echo "  Pipeline complete."
echo "  SFT checkpoint: $HF_MODEL_PATH"
echo "  RL output: $PROJECT_DIR/outputs/revise_nextqa_grpo_after_sft"
echo "================================================================"
