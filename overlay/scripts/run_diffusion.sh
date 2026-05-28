#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# run_diffusion.sh — 一键启动 Diffusion Policy 部署
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot}"

# Diffusion 支持两种加载方式:
#   1) CKPT_DIR: HuggingFace format 目录 (包含 config.json + model.safetensors)
#   2) CKPT_SD:  裸 state_dict .pt 文件 (需要手动指定 state_dim/action_dim)
CKPT_DIR="${CKPT_DIR:-}"
CKPT_SD="${CKPT_SD:-}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
CAM_SIDE="${CAM_SIDE:-0}"
RS_SERIAL="${RS_SERIAL:-243522072793}"
MAX_STEPS="${MAX_STEPS:-1000}"
DEVICE="${DEVICE:-cuda}"
MAX_REL="${MAX_REL:-10.0}"
STATE_DIM="${STATE_DIM:-6}"
ACTION_DIM="${ACTION_DIM:-6}"

echo "============================================"
echo "  Diffusion Policy Deployment"
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

# 确定 checkpoint 参数
if [[ -n "$CKPT_DIR" ]]; then
    [[ -d "$CKPT_DIR" ]] || { echo "ERROR: Directory not found: $CKPT_DIR"; exit 1; }
    echo "Checkpoint: $CKPT_DIR (HF format)"
    CKPT_ARGS="--ckpt_dir $CKPT_DIR"
elif [[ -n "$CKPT_SD" ]]; then
    [[ -f "$CKPT_SD" ]] || { echo "ERROR: File not found: $CKPT_SD"; exit 1; }
    echo "Checkpoint: $CKPT_SD (state_dict)"
    CKPT_ARGS="--ckpt_state_dict $CKPT_SD --state_dim $STATE_DIM --action_dim $ACTION_DIM"
else
    echo "ERROR: Set either CKPT_DIR or CKPT_SD"
    echo "  CKPT_DIR=path/to/hf_dir bash $0"
    echo "  CKPT_SD=path/to/model.pt bash $0"
    exit 1
fi

echo "Safety:     max_relative_target=${MAX_REL}°/step"
echo "============================================"

# shellcheck disable=SC2086
exec python deploy/deploy_diffusion.py \
    $CKPT_ARGS \
    --robot_port "$ROBOT_PORT" \
    --cam_side "$CAM_SIDE" \
    --realsense_serial "$RS_SERIAL" \
    --max_steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --max_relative_target "$MAX_REL"
