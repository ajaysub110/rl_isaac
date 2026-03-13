#!/usr/bin/env bash
# train.sh — Launch RSL_RL PPO training on Isaac-Humanoid-v0
#
# Benchmark config: 4096 envs × 32 rollout steps × 500 iterations = 65.5M steps
# Expected time: ~198s on a single RTX 4090
#
# Usage:
#   ./train.sh                    # full benchmark run
#   ./train.sh --max_iterations 2 # quick smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"

# Defaults matching the official benchmark
NUM_ENVS=4096
MAX_ITERATIONS=500
EXTRA_ARGS=()

# Parse overrides — any flag recognized here is consumed, the rest pass through
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_envs)     NUM_ENVS="$2";       shift 2 ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1");  shift ;;
    esac
done

echo "=== RSL_RL PPO Training ==="
echo "  Task:           Isaac-Humanoid-v0"
echo "  Environments:   ${NUM_ENVS}"
echo "  Iterations:     ${MAX_ITERATIONS}"
echo "  Total steps:    $(( NUM_ENVS * 32 * MAX_ITERATIONS ))"
echo "  Logs:           logs/rsl_rl/humanoid_direct/"
echo ""

cd "${SCRIPT_DIR}"
python "${TRAIN_SCRIPT}" \
    --task Isaac-Humanoid-v0 \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
