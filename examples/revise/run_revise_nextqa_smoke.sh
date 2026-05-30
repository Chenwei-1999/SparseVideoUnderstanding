#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke run for REVISE NExT-QA using 4 GPUs and tiny sample set.
# Usage:
#   ENGINE=sglang ./examples/revise/run_revise_nextqa_smoke.sh
#   ENGINE=vllm   ./examples/revise/run_revise_nextqa_smoke.sh

ENGINE=${ENGINE:-sglang}
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/revise/config"

# Respect existing CUDA_VISIBLE_DEVICES; default to 0,1,2,3
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES

python3 -m verl.trainer.main_ppo \
  --config-path "$CONFIG_PATH" \
  --config-name revise_nextqa_smoke \
  actor_rollout_ref.rollout.name="$ENGINE" \
  trainer.logger='["console"]' \
  "$@"

