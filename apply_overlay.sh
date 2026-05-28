#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/lerobot"
    exit 1
fi

TARGET_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_DIR="$SCRIPT_DIR/overlay"

if [[ ! -d "$TARGET_ROOT/.git" ]]; then
    echo "ERROR: target does not look like a git checkout: $TARGET_ROOT"
    exit 1
fi

if [[ ! -d "$TARGET_ROOT/src/lerobot" ]]; then
    echo "ERROR: target does not look like a LeRobot checkout: $TARGET_ROOT"
    exit 1
fi

rsync -av --exclude '__pycache__' "$OVERLAY_DIR/" "$TARGET_ROOT/"

echo "Overlay applied to: $TARGET_ROOT"
echo "Next steps:"
echo "  1. cd $TARGET_ROOT"
echo "  2. inspect git diff"
echo "  3. configure ports, calibration paths, and checkpoints"