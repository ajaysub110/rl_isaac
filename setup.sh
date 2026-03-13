#!/usr/bin/env bash
# setup.sh — Create conda env and install Isaac Lab + RSL_RL
set -euo pipefail

ENV_NAME="rl_isaac_env"
CONDA="/opt/miniforge3/bin/conda"
ISAAC_SIM_VERSION="5.1.0"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/6] Creating conda environment '${ENV_NAME}' (Python 3.11) ==="
"$CONDA" create -n "$ENV_NAME" python=3.11 -y

# Helper to run commands inside the conda env
run_in_env() {
    "$CONDA" run -n "$ENV_NAME" --no-capture-output "$@"
}

echo "=== [2/6] Upgrading pip ==="
run_in_env pip install --upgrade pip

echo "=== [3/6] Installing Isaac Sim ${ISAAC_SIM_VERSION} ==="
run_in_env pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

echo "=== [4/6] Installing PyTorch ${TORCH_VERSION}+cu128 ==="
run_in_env pip install -U "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== [5/6] Cloning & installing Isaac Lab with RSL_RL ==="
sudo apt-get update -qq && sudo apt-get install -y -qq cmake build-essential
if [ ! -d "${SCRIPT_DIR}/IsaacLab" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "${SCRIPT_DIR}/IsaacLab"
fi

cd "${SCRIPT_DIR}/IsaacLab"
run_in_env ./isaaclab.sh --install rsl_rl

echo "=== [6/6] Installing TensorBoard ==="
run_in_env pip install tensorboard

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate with: conda activate ${ENV_NAME}"
echo "============================================"
