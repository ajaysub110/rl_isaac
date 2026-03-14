# Unitree H1 Motion Imitation (RL-IsaacLab)

This repository contains an end-to-end pipeline for training a Unitree H1 humanoid robot to imitate human motion capture data (AMASS) using Reinforcement Learning within NVIDIA's Isaac Sim/IsaacLab.

## 🚀 The Pipeline

1. **AMASS (Mocap Data)**: Raw 3D human pose sequences (SMPL parametric format).
2. **ASAP Retargeting**: Kinematic retargeting from the human SMPL model to the H1 skeleton (19 DOFs).
3. **IsaacLab Direct Task**: An RSL_RL PPO policy that learns to track the retargeted reference motion dynamically while maintaining balance.

---

## 🛠️ Prerequisites & Dependencies

Several core dependencies are **not included** in this codebase and must be downloaded/installed separately:
*   **Operating System**: Linux (Ubuntu 22.04 or 24.04 recommended)
*   **GPU**: NVIDIA RTX GPU (RTX 4090 recommended)
*   [**Isaac Sim 5.1.0**](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html)
*   [**IsaacLab**](https://github.com/isaac-sim/IsaacLab) (Included as a git submodule)
*   [**ASAP Pipeline**](https://github.com/LeCAR-Lab/ASAP) (Aligning Simulation and Real-World Physics)
*   **SMPL Models**: Necessary for ASAP retargeting (`SMPL_python_v.1.1.0.zip`).
*   [**AMASS Dataset**](https://amass.is.tue.mpg.de/): Human mocap dataset (`.npz` files).

---

## ⚙️ Setup & Installation

### 1. Create Conda Environment
A fully configured `environment.yaml` is provided. This installs Python 3.11, PyTorch (CUDA), Isaac Sim via pip, and other math dependencies.
```bash
conda env create -f environment.yaml
conda activate h1_mimic_env
```
*(Alternatively, you can run the provided `./setup.sh` to construct the environment manually.)*

### 2. Install IsaacLab and RSL_RL
Initialize the IsaacLab submodule and install its extensions + `rsl_rl`:
```bash
git submodule update --init --recursive
cd IsaacLab
./isaaclab.sh --install rsl_rl
cd ..
```

### 3. Setup ASAP (for Retargeting)
To run retargeting, clone the ASAP repository alongside this one.
```bash
git clone https://github.com/LeCAR-Lab/ASAP.git ../ASAP
cd ../ASAP

# Install ASAP modules (smpl_sim and humanoidverse)
uv pip install -e smpl_sim/
uv pip install -e .
```

*Note for Python 3.11 users: You may need to patch the `chumpy` library by replacing `inspect.getargspec` with `inspect.getfullargspec` in `chumpy/ch.py` to fix a deprecation crash.*

### 4. Provide SMPL & AMASS Data
ASAP relies on the SMPL body model and AMASS motion parameters.
1. Create a `data/smpl` folder inside the ASAP directory.
2. Download the basic SMPL `.pkl` files (male, female, neutral) from the official SMPL website and place them inside `data/smpl`.
3. Download AMASS `.npz` motions and place them in the appropriate directory for ASAP's shape/motion fitting scripts.

---

## 🤸 Running the Pipeline

### Step 1: Retargeting (ASAP)
Use ASAP to fit the SMPL shape to the H1 robot, and then kinematically retarget the AMASS motion file to the H1's 19 joints.
```bash
cd ../ASAP
# 1. Optimize shape scale to H1
python scripts/data_process/fit_smpl_shape.py +robot=h1/h1_19dof

# 2. Fit the motion frames
python scripts/data_process/fit_smpl_motion.py +robot=h1/h1_19dof
```
This produces an optimized `.pkl` file. We use a custom conversion script to extract `body_rotations` as `wxyz` quaternions and save the final data to `outputs/h1_roundhouse.npz` within the `rl_isaac` directory.

---

### Step 2: Policy Training (IsaacLab)
The `Isaac-H1-Mimic-Direct-v0` task trains the robot using RSL_RL PPO.
The reward structure (DeepMimic-style) penalizes tracking errors while explicitly rewarding stability (`alive` and `upright` bonuses). The policy learns to output **residual joint actions** around the reference motion limits.

Launch the training using the provided shell script (default: 4096 envs, 3000 iterations):
```bash
cd rl_isaac
./train_mimic.sh
```

To monitor training curves, launch TensorBoard in a separate terminal:
```bash
tensorboard --logdir logs/rsl_rl/h1_mimic_direct/
```

---

### Step 3: Evaluation & Video Generation
Once trained, use the evaluation script to visualize the policy or render it headlessly to an MP4 video.
```bash
# Render headless video (spins up Xvfb automatically) and record 300 steps
./eval_mimic.sh --video --video_length 300

# (Optional) Evaluate a specific previous checkpoint folder:
./eval_mimic.sh --video --video_length 300 --load_run 2026-03-14_01-48-18
```
The robot model includes a custom metallic blue SDK override in `_setup_scene()` to enhance visual clarity in evaluation videos. 

---

## 📝 Project Architecture Notes
*   **Root Body Initialization**: The environment uses `pelvis` as the `reference_body`. Correcting this from `torso_link` prevents the robot from sinking/floating on env reset due to kinematic offsets.
*   **Balance vs. Tracking**: Retargeted kinematic motions are often not dynamically feasible on a real robot. The added `upright bonus` helps the RL agent shift its center of mass appropriately during athletic movements (like kicks) without falling.
*   **Residual Actions vs. Absolute Tracking**: Outputting residuals (±1.0 rad) around the ground-truth reference joint state is dramatically easier for PPO to learn than a raw `(-π, π)` action space.
