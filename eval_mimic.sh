#!/usr/bin/env bash
# eval_mimic.sh — Run a trained RSL_RL policy on Isaac-H1-Mimic-Direct-v0
#
# Usage:
#   ./eval_mimic.sh                              # play latest checkpoint (headless)
#   ./eval_mimic.sh --video                      # record video of the policy
#   ./eval_mimic.sh --video --video_length 300   # record longer video
#   ./eval_mimic.sh --load_run 2026-03-13_19-30  # play a specific run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAY_SCRIPT="${SCRIPT_DIR}/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py"

NUM_ENVS=1
EXTRA_ARGS=()
USE_VIDEO=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_envs)  NUM_ENVS="$2";      shift 2 ;;
        --video)     USE_VIDEO=true;     EXTRA_ARGS+=("$1"); shift ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== H1 Mimic Policy Evaluation ==="
echo "  Task:         Isaac-H1-Mimic-Direct-v0"
echo "  Environments: ${NUM_ENVS}"
echo ""

# When recording video, start a virtual display (Xvfb) for offscreen rendering
if $USE_VIDEO; then
    echo "[INFO] Starting virtual display (Xvfb) for video rendering..."
    Xvfb :99 -screen 0 1280x720x24 &>/dev/null &
    XVFB_PID=$!
    sleep 1
    export DISPLAY=:99
    trap "kill $XVFB_PID 2>/dev/null || true" EXIT
fi

cd "${SCRIPT_DIR}"
python "${PLAY_SCRIPT}" \
    --task Isaac-H1-Mimic-Direct-v0 \
    --num_envs "${NUM_ENVS}" \
    --headless \
    --enable_cameras \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
