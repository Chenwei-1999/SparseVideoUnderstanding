#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper for the paper-suite NExT-QA RL-after-SFT row.
# Prefer calling scripts/paper_suite.py directly in new automation.

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/repro_runs/table4_nextqa}"

"$PYTHON_BIN" scripts/paper_suite.py run \
  --experiment nextqa_table4_rl_after_sft \
  --output-dir "$OUTPUT_DIR" \
  "$@"
