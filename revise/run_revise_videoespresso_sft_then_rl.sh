#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/revise/config"
ENGINE="${ENGINE:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SFT_CKPT_DIR="${SFT_CKPT_DIR:-$PROJECT_DIR/outputs/revise_videoespresso_sft}"
TEACHER_LOG="${TEACHER_LOG:-$PROJECT_DIR/outputs/videoespresso_teacher_train_log.jsonl}"
read -r -a SFT_EXTRA_ARGS_ARRAY <<< "${SFT_EXTRA_ARGS:-}"
read -r -a RL_EXTRA_ARGS_ARRAY <<< "${RL_EXTRA_ARGS:-}"

if [ ! -f "$TEACHER_LOG" ]; then
    echo "ERROR: Teacher data log not found at $TEACHER_LOG"
    echo "Run ./revise/run_generate_teacher_data_videoespresso.sh first."
    exit 1
fi

SFT_INPUT="$TEACHER_LOG" ./revise/run_revise_videoespresso_sft.sh "${SFT_EXTRA_ARGS_ARRAY[@]}"

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
    echo "ERROR: HF model not found under $LATEST_STEP"
    exit 1
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path "$CONFIG_PATH" \
  --config-name revise_videoespresso_grpo_after_sft \
  actor_rollout_ref.model.path="$HF_MODEL_PATH" \
  actor_rollout_ref.rollout.name="$ENGINE" \
  trainer.logger='["console","wandb"]' \
  "${RL_EXTRA_ARGS_ARRAY[@]}" \
  "$@"
