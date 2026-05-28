#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# run_vla_only.sh — 一键启动 SmolVLA（无触觉）部署
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot}"

CKPT="${CKPT:-train/ckpt_stage4_smolvla_final.pt}"
TASK="${TASK:-grasp the cloth from the box}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
CAM_SIDE="${CAM_SIDE:-0}"
RS_SERIAL="${RS_SERIAL:-243522072793}"
MAX_STEPS="${MAX_STEPS:-1000}"
DEVICE="${DEVICE:-cuda}"
MAX_REL="${MAX_REL:-10.0}"

echo "============================================"
echo "  SmolVLA (no tactile) Deployment"
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

python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA:       {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available')
"

[[ -f "$CKPT" ]] || { echo "ERROR: Checkpoint not found: $CKPT"; exit 1; }
echo "Checkpoint: $CKPT"
echo "Task:       $TASK"
echo "Safety:     max_relative_target=${MAX_REL}°/step"
echo "============================================"

exec python deploy/deploy_vla_only.py \
    --ckpt "$CKPT" \
    --task "$TASK" \
    --robot_port "$ROBOT_PORT" \
    --cam_side "$CAM_SIDE" \
    --realsense_serial "$RS_SERIAL" \
    --max_steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --max_relative_target "$MAX_REL"
