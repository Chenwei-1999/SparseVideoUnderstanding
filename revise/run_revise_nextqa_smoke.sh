#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible smoke wrapper for the paper-suite NExT-QA PnP row.
# Prefer calling scripts/paper_suite.py directly in new automation.

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/repro_runs/table4_nextqa_smoke}"

"$PYTHON_BIN" scripts/paper_suite.py run \
  --experiment nextqa_table4_pnp \
  --smoke \
  --output-dir "$OUTPUT_DIR" \
  "$@"
