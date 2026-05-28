#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# run_eval.sh — 一键启动统一评测
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot}"

MODEL="${MODEL:-diffusion}"         # tactile_vla / vla_only / diffusion
CKPT="${CKPT:-train/ckpt_onlydiffusion_ep40.pt}"
TASK="${TASK:-grasp the cloth from the box}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
RS_SERIAL="${RS_SERIAL:-243522072793}"
DEVICE="${DEVICE:-cuda}"
MAX_REL="${MAX_REL:-15.0}"
N_EPISODES="${N_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-900}"
OUTPUT="${OUTPUT:-eval_results.json}"
TACTILE_MOCK="${TACTILE_MOCK:-}"
TACTILE_LEFT_PORT="${TACTILE_LEFT_PORT:-/dev/ttyUSB0}"
TACTILE_RIGHT_PORT="${TACTILE_RIGHT_PORT:-/dev/ttyUSB1}"
TACTILE_BAUDRATE="${TACTILE_BAUDRATE:-115200}"
FPS="${FPS:-30}"                       # 与采数据 fps 对齐
SKIP_CAL="${SKIP_CAL:-0}"               # 1=跳过标定（复用采数据时的标定文件）
FOLLOWER_CALIBRATION_FILE="${FOLLOWER_CALIBRATION_FILE:-}"
RECORD="${RECORD:-0}"                   # 1=录制评测数据集（用于回看视频）
RECORD_REPO="${RECORD_REPO:-natsuu/eval_replay}"

echo "============================================"
echo "  SO-101 Evaluation"
echo "============================================"

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Conda env:  $CONDA_DEFAULT_ENV"
else
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
    fi
fi

python -c "
import lerobot; v = lerobot.__version__
print(f'LeRobot:    v{v}')
assert v.startswith('0.5'), f'Expected lerobot 0.5.x, got {v}'
" || { echo "ERROR: lerobot version check failed."; exit 1; }

echo "Model:      $MODEL"
echo "Checkpoint: $CKPT"
echo "Task:       $TASK"
echo "FPS:        $FPS"
echo "Episodes:   $N_EPISODES"
echo "Max steps:  $MAX_STEPS/episode"
echo "Skip cal:   $SKIP_CAL"
echo "Cal file:   ${FOLLOWER_CALIBRATION_FILE:-default}"
echo "Record:     $RECORD"
echo "Output:     $OUTPUT"
echo "============================================"

CMD=(
    python deploy/eval_real_robot.py
    --model "$MODEL"
    --ckpt "$CKPT"
    --task "$TASK"
    --robot_port "$ROBOT_PORT"
    --realsense_serial "$RS_SERIAL"
    --device "$DEVICE"
    --max_relative_target "$MAX_REL"
    --n_episodes "$N_EPISODES"
    --max_steps_per_episode "$MAX_STEPS"
    --output_json "$OUTPUT"
    --fps "$FPS"
    --tactile_left_port "$TACTILE_LEFT_PORT"
    --tactile_right_port "$TACTILE_RIGHT_PORT"
    --tactile_baudrate "$TACTILE_BAUDRATE"
)

if [[ -n "$TACTILE_MOCK" ]]; then
    CMD+=(--tactile_mock)
fi

if [[ -n "$FOLLOWER_CALIBRATION_FILE" ]]; then
    CMD+=(--follower_calibration_file "$FOLLOWER_CALIBRATION_FILE")
fi

if [[ "$SKIP_CAL" == "1" ]]; then
    CMD+=(--skip_calibration)
fi

if [[ "$RECORD" == "1" ]]; then
    CMD+=(--record_dataset --record_repo_id "$RECORD_REPO")
fi

exec "${CMD[@]}"
