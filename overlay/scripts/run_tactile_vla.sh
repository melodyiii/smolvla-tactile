#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# run_tactile_vla.sh — 一键启动 SmolVLA + 触觉传感器部署
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot}"

# ─── 默认参数（可通过环境变量覆盖） ───
CKPT="${CKPT:-train/ckpt_stage4_smolvla_final.pt}"
TASK="${TASK:-grasp the cloth from the box}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
CAM_SIDE="${CAM_SIDE:-0}"
RS_SERIAL="${RS_SERIAL:-243522072793}"
MAX_STEPS="${MAX_STEPS:-1000}"
DEVICE="${DEVICE:-cuda}"
MAX_REL="${MAX_REL:-10.0}"
TACTILE_MOCK="${TACTILE_MOCK:-}"  # 设为 1 启用 mock
TACTILE_LEFT_PORT="${TACTILE_LEFT_PORT:-/dev/ttyUSB0}"
TACTILE_RIGHT_PORT="${TACTILE_RIGHT_PORT:-/dev/ttyUSB1}"
TACTILE_BAUDRATE="${TACTILE_BAUDRATE:-115200}"

# ─── 环境检查 ───
echo "============================================"
echo "  SmolVLA + Tactile Deployment"
echo "============================================"

# 检查 conda 环境
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Conda env:  $CONDA_DEFAULT_ENV"
else
    echo "WARNING: No conda env active. Trying to activate '$CONDA_ENV_NAME'..."
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate "$CONDA_ENV_NAME" 2>/dev/null || echo "  Failed to activate '$CONDA_ENV_NAME'. Proceeding with current env."
    fi
fi

# 检查 lerobot 版本
python -c "
import lerobot; v = lerobot.__version__
print(f'LeRobot:    v{v}')
assert v.startswith('0.5'), f'Expected lerobot 0.5.x, got {v}. Check your environment.'
" || { echo "ERROR: lerobot version check failed."; exit 1; }

# 检查 CUDA
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA:       {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available, using CPU (will be slow)')
"

# 检查 checkpoint
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
echo "Checkpoint: $CKPT"
echo "Task:       $TASK"
echo "Robot port: $ROBOT_PORT"
echo "Safety:     max_relative_target=${MAX_REL}°/step"
echo "============================================"

# ─── 构建命令 ───
CMD=(
    python deploy/deploy_tactile_vla.py
    --ckpt "$CKPT"
    --task "$TASK"
    --robot_port "$ROBOT_PORT"
    --cam_side "$CAM_SIDE"
    --realsense_serial "$RS_SERIAL"
    --max_steps "$MAX_STEPS"
    --device "$DEVICE"
    --max_relative_target "$MAX_REL"
    --tactile_left_port "$TACTILE_LEFT_PORT"
    --tactile_right_port "$TACTILE_RIGHT_PORT"
    --tactile_baudrate "$TACTILE_BAUDRATE"
)

if [[ -n "$TACTILE_MOCK" ]]; then
    CMD+=(--tactile_mock)
    echo "NOTE: Using MOCK tactile data (no hardware)"
fi

echo ""
echo "Running: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
