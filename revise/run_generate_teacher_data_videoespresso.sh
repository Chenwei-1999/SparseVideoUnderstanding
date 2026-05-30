#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ASSET_ROOT="${REVISE_ASSET_ROOT:-$PROJECT_DIR/data/revise_assets}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_PATH:-${REVISE_QWEN25_VL_7B_PATH:-$ASSET_ROOT/models/Qwen2.5-VL-7B-Instruct}}}"
TEACHER_MODEL_ID="${TEACHER_MODEL_ID:-${MODEL_ID:-}}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-${BASE_URL:-}}"
VIDEO_ROOT="${VIDEO_ROOT:-${REVISE_VIDEOESPRESSO_ROOT:-$ASSET_ROOT/VideoEspresso}}"
JSON="${JSON:-$PROJECT_DIR/outputs/videoespresso_train_mc.json}"
MAX_SAMPLES="${MAX_SAMPLES:-8000}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
LOG_PATH="${LOG_PATH:-$PROJECT_DIR/outputs/videoespresso_teacher_train_log.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
SERVER_LOG="${SERVER_LOG:-}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-1800}"

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
    if [ -n "$TEACHER_MODEL_ID" ]; then
        MODEL_ARGS+=(--model-id "$TEACHER_MODEL_ID")
    fi
    if [ -n "$SERVER_LOG" ]; then
        mkdir -p "$(dirname "$SERVER_LOG")"
        MODEL_ARGS+=(--server-log "$SERVER_LOG")
    fi
fi

if [ ! -f "$JSON" ]; then
    echo "ERROR: MC train JSON not found at $JSON"
    echo "Run: python scripts/prepare_videoespresso_mc_train.py --output $JSON"
    exit 1
fi

echo "=== Generating VideoEspresso teacher data ==="
if [ -n "$TEACHER_BASE_URL" ]; then
    echo "  Teacher API: $TEACHER_BASE_URL"
    echo "  Teacher id:  $TEACHER_MODEL_ID"
else
    echo "  Teacher:     $TEACHER_MODEL_PATH"
    if [ -n "$TEACHER_MODEL_ID" ]; then
        echo "  Teacher id:  $TEACHER_MODEL_ID"
    fi
fi
echo "  JSON:        $JSON"
echo "  Max samples: $MAX_SAMPLES"
echo "  Log:         $LOG_PATH"
if [ -z "$TEACHER_BASE_URL" ] && [ -n "$SERVER_LOG" ]; then
    echo "  Server log:  $SERVER_LOG"
fi
echo "  Shards:      $NUM_SHARDS (idx=$SHARD_IDX)"

"$PYTHON_BIN" "$PROJECT_DIR/revise/plug_and_play_egoschema_vllm.py" \
    "${MODEL_ARGS[@]}" \
    --dataset-name videoespresso \
    --json "$JSON" \
    --video-root "$VIDEO_ROOT" \
    --max-samples "$MAX_SAMPLES" \
    --max-rounds 5 \
    --max-frames-per-round 5 \
    --log-jsonl "$LOG_PATH" \
    --num-shards "$NUM_SHARDS" \
    --shard-idx "$SHARD_IDX" \
    --resume-from-log \
    "$@"
