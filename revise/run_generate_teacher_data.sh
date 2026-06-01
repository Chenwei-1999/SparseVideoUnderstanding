#!/usr/bin/env bash
set -euo pipefail

# Generate SFT teacher data by running the configured teacher on NExT-QA TRAIN split.
#
# This avoids data leakage: SFT trains on train-split teacher outputs,
# evaluation uses val/test split.
#
# Usage:
#   ./revise/run_generate_teacher_data.sh [extra plug_and_play args...]
#
# Environment variables:
#   TEACHER_MODEL_PATH — local/HF model path for the teacher server
#                        (default: local Qwen2.5-VL-7B fallback)
#   TEACHER_MODEL_ID   — optional OpenAI-compatible served model id
#   TEACHER_BASE_URL   — optional external OpenAI-compatible endpoint
#   MODEL_PATH/MODEL_ID/BASE_URL are accepted as backwards-compatible aliases.
#   VIDEO_ROOT  — NExT-QA video root
#   MAP_JSON    — vid → vidorID mapping
#   CSV         — NExT-QA CSV split to use (default: train.csv)
#   MAX_SAMPLES — max samples (default: 8000, matching the paper-scale SFT set)
#   MAX_ROUNDS — max REVISE rounds (default: 4, matching NExT-QA Table 4)
#   MAX_FRAMES_PER_ROUND — max frames per select round (default: 3, matching Table 4)
#   NUM_SHARDS  — for multi-GPU data-parallel (default: 1)
#   SHARD_IDX   — shard index (default: 0)
#   LOG_PATH    — output JSONL path
#   PORT        — vLLM server port for this shard (default: 18000)

PROJECT_DIR="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ASSET_ROOT="${REVISE_ASSET_ROOT:-$PROJECT_DIR/data/revise_assets}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_PATH:-${REVISE_QWEN25_VL_7B_PATH:-$ASSET_ROOT/models/Qwen2.5-VL-7B-Instruct}}}"
TEACHER_MODEL_ID="${TEACHER_MODEL_ID:-${MODEL_ID:-}}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-${BASE_URL:-}}"
VIDEO_ROOT="${VIDEO_ROOT:-${REVISE_NEXTQA_VIDEO_ROOT:-$ASSET_ROOT/NExT-QA/NExTVideo}}"
MAP_JSON="${MAP_JSON:-${REVISE_NEXTQA_MAP_JSON:-$ASSET_ROOT/NExT-QA/map_vid_vidorID.json}}"
CSV="${CSV:-${REVISE_NEXTQA_TRAIN_CSV:-$ASSET_ROOT/NExT-QA/nextqa/train.csv}}"
MAX_SAMPLES="${MAX_SAMPLES:-8000}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
MAX_FRAMES_PER_ROUND="${MAX_FRAMES_PER_ROUND:-3}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
LOG_PATH="${LOG_PATH:-$PROJECT_DIR/outputs/nextqa_teacher_train_log.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
SERVER_LOG="${SERVER_LOG:-}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-1800}"
PORT="${PORT:-18000}"

MODEL_ARGS=()
if [ -n "$TEACHER_BASE_URL" ]; then
    if [ -z "$TEACHER_MODEL_ID" ]; then
        echo "ERROR: TEACHER_BASE_URL requires TEACHER_MODEL_ID."
        exit 1
    fi
    MODEL_ARGS+=(--model-path "$TEACHER_MODEL_ID" --base-url "$TEACHER_BASE_URL" --model-id "$TEACHER_MODEL_ID")
else
    MODEL_ARGS+=(--model-path "$TEACHER_MODEL_PATH" --start-server)
    MODEL_ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
    MODEL_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    MODEL_ARGS+=(--server-timeout-s "$SERVER_TIMEOUT_S")
    MODEL_ARGS+=(--port "$PORT")
    if [ -n "$TEACHER_MODEL_ID" ]; then
        MODEL_ARGS+=(--model-id "$TEACHER_MODEL_ID")
    fi
    if [ -n "$SERVER_LOG" ]; then
        mkdir -p "$(dirname "$SERVER_LOG")"
        MODEL_ARGS+=(--server-log "$SERVER_LOG")
    fi
fi

echo "=== Generating teacher data ==="
if [ -n "$TEACHER_BASE_URL" ]; then
    echo "  Teacher API: $TEACHER_BASE_URL"
    echo "  Teacher id:  $TEACHER_MODEL_ID"
else
    echo "  Teacher:     $TEACHER_MODEL_PATH"
    if [ -n "$TEACHER_MODEL_ID" ]; then
        echo "  Teacher id:  $TEACHER_MODEL_ID"
    fi
fi
echo "  CSV:         $CSV"
echo "  Max samples: $MAX_SAMPLES"
echo "  Max rounds:  $MAX_ROUNDS"
echo "  Max frames:  $MAX_FRAMES_PER_ROUND"
echo "  Log:         $LOG_PATH"
if [ -z "$TEACHER_BASE_URL" ] && [ -n "$SERVER_LOG" ]; then
    echo "  Server log:  $SERVER_LOG"
fi
if [ -z "$TEACHER_BASE_URL" ]; then
    echo "  Server port: $PORT"
fi
echo "  Shards:      $NUM_SHARDS (idx=$SHARD_IDX)"

"$PYTHON_BIN" "$PROJECT_DIR/revise/pnp_cli.py" \
    --dataset nextqa \
    --backend vllm_http \
    --setting multi_round_pnp \
    "${MODEL_ARGS[@]}" \
    --video-root "$VIDEO_ROOT" \
    --map-json "$MAP_JSON" \
    --csv "$CSV" \
    --max-samples "$MAX_SAMPLES" \
    --max-rounds "$MAX_ROUNDS" \
    --max-frames-per-round "$MAX_FRAMES_PER_ROUND" \
    --log-jsonl "$LOG_PATH" \
    --num-shards "$NUM_SHARDS" \
    --shard-idx "$SHARD_IDX" \
    --resume-from-log \
    "$@"

echo "=== Teacher data generation complete ==="
echo "  Log: $LOG_PATH"
echo ""
echo "Next step: generate SFT parquet data:"
echo "  python revise/generate_sft_data.py --input $LOG_PATH --output outputs/sft_data/revise_sft.parquet"
