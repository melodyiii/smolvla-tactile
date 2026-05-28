#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$SCRIPT_DIR}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot}"

EPISODE_TIME_S="${EPISODE_TIME_S:-20}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so101_follower_arm}"
ROBOT_CALIB_DIR="${ROBOT_CALIB_DIR:-$LEROBOT_ROOT/calibration/robots/so101_follower}"
TACTILE_LEFT_PORT="${TACTILE_LEFT_PORT:-/dev/ttyUSB0}"
TACTILE_RIGHT_PORT="${TACTILE_RIGHT_PORT:-/dev/ttyUSB1}"
TACTILE_BAUDRATE="${TACTILE_BAUDRATE:-115200}"
RS_SERIAL="${RS_SERIAL:-243522072793}"
SIDE_CAM_INDEX="${SIDE_CAM_INDEX:-0}"
TELEOP_PORT="${TELEOP_PORT:-/dev/ttyACM1}"
TELEOP_ID="${TELEOP_ID:-so101_leader_arm}"
TELEOP_CALIB_DIR="${TELEOP_CALIB_DIR:-$LEROBOT_ROOT/calibration/teleoperators/so101_leader}"
DATASET_REPO_ID="${DATASET_REPO_ID:-your_hf_user/so101_tactile_dataset}"
TASK_NAME="${TASK_NAME:-Pick and place with tactile feedback}"
NUM_EPISODES="${NUM_EPISODES:-10}"
DISPLAY_IP="${DISPLAY_IP:-127.0.0.1}"
DISPLAY_PORT="${DISPLAY_PORT:-9876}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
fi

if [[ ! -d "$LEROBOT_ROOT/src/lerobot" ]]; then
  echo "ERROR: Set LEROBOT_ROOT to your patched LeRobot checkout. Current value: $LEROBOT_ROOT"
  exit 1
fi

beep() {
  aplay -q /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null || printf '\a'
}

cd "$LEROBOT_ROOT"

lerobot-record \
  --robot.type=so101_tactile \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.calibration_dir="$ROBOT_CALIB_DIR" \
  --robot.tactile_left.port="$TACTILE_LEFT_PORT" \
  --robot.tactile_left.baudrate="$TACTILE_BAUDRATE" \
  --robot.tactile_right.port="$TACTILE_RIGHT_PORT" \
  --robot.tactile_right.baudrate="$TACTILE_BAUDRATE" \
  --robot.cameras="{
    depth: {type: intelrealsense, serial_number_or_name: \"$RS_SERIAL\", width: 640, height: 480, fps: 30},
    side: {type: opencv, index_or_path: $SIDE_CAM_INDEX, width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id="$TELEOP_ID" \
  --teleop.calibration_dir="$TELEOP_CALIB_DIR" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.single_task="$TASK_NAME" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.fps=30 \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=false \
  --dataset.vcodec=h264 \
  --dataset.encoder_threads=2 \
  --display_data=true \
  --display_ip="$DISPLAY_IP" \
  --display_port="$DISPLAY_PORT" \
  --play_sounds=false \
  2>&1 | tee -a /tmp/lerobot_record.log | while IFS= read -r line; do
    echo "$line"

    if [[ "$line" == *"Recording episode"* ]]; then
      echo "[PROMPT] Recording started, begin teleoperation now."
      beep

      (
        pre=$((EPISODE_TIME_S - 5))
        if [[ "$pre" -gt 0 ]]; then
          sleep "$pre"
        fi
        for s in 5 4 3 2 1; do
          echo "[PROMPT] ${s}s to stop recording..."
          beep
          sleep 1
        done
      ) &
    fi
  done
