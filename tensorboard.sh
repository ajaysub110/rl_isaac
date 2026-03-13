#!/usr/bin/env bash
# tensorboard.sh — Launch TensorBoard for training logs
#
# Usage:
#   ./tensorboard.sh           # default port 6006
#   ./tensorboard.sh 6007      # custom port
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/rsl_rl"
PORT="${1:-6006}"

echo "=== TensorBoard ==="
echo "  Log dir: ${LOG_DIR}"
echo "  URL:     http://localhost:${PORT}"
echo ""

tensorboard --logdir "${LOG_DIR}" --port "${PORT}" --bind_all
