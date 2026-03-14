#!/usr/bin/env bash
# train_mimic.sh — Launch RSL_RL PPO training on Isaac-H1-Mimic-Direct-v0
#
# DeepMimic-style imitation reward for H1 roundhouse kick
#
# Usage:
#   ./train_mimic.sh                        # full training run
#   ./train_mimic.sh --max_iterations 2     # quick smoke test
#   ./train_mimic.sh --num_envs 64          # small env count for debugging
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"

# Defaults
NUM_ENVS=4096
MAX_ITERATIONS=3000
EXTRA_ARGS=()

# Parse overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_envs)       NUM_ENVS="$2";       shift 2 ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        *)                EXTRA_ARGS+=("$1");  shift ;;
    esac
done

echo "=== H1 Mimic Training (RSL_RL PPO) ==="
echo "  Task:           Isaac-H1-Mimic-Direct-v0"
echo "  Environments:   ${NUM_ENVS}"
echo "  Iterations:     ${MAX_ITERATIONS}"
echo "  Total steps:    $(( NUM_ENVS * 24 * MAX_ITERATIONS ))"
echo "  Logs:           logs/rsl_rl/h1_mimic_direct/"
echo ""

cd "${SCRIPT_DIR}"
python "${TRAIN_SCRIPT}" \
    --task Isaac-H1-Mimic-Direct-v0 \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
