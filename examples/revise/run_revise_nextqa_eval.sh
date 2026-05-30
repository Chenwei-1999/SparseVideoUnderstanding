#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ENGINE=vllm ./examples/revise/run_revise_nextqa_eval.sh

ENGINE=${ENGINE:-sglang}
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/revise/config"

python3 -m verl.trainer.main_ppo \
  --config-path "$CONFIG_PATH" \
  --config-name revise_nextqa_eval \
  actor_rollout_ref.rollout.name="$ENGINE" \
  "$@"
