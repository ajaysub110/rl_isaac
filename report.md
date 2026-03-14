# Motion Imitation for Unitree H1: Method & Key Insights

## Overview

This project trains a Unitree H1 humanoid robot to imitate a roundhouse kick from human motion capture data, using reinforcement learning in IsaacLab. The full pipeline runs end-to-end: raw AMASS mocap → SMPL body model → retargeting to H1 kinematics → RL policy training → physical simulation.

## Pipeline

```
AMASS NPZ (SMPL poses) ──► ASAP Retargeting ──► H1 Joint Angles ──► RSL_RL PPO ──► Kick Policy
     100 frames                shape+motion        19 DOFs             4096 envs       balanced
     30 fps                    fitting              + body poses        3000 iters      kick
```

## Stage 1: Motion Data

**Input**: AMASS NPZ file containing a roundhouse kick (100 frames @ 30fps, 3.3s). This encodes the motion as SMPL body model parameters — axis-angle rotations for 24 body joints plus root translation.

## Stage 2: Retargeting via ASAP

We use [ASAP](https://github.com/LeCAR-Lab/ASAP) (Aligning Simulation and Real-World Physics) for retargeting. This is a PyTorch-based pipeline that:

1. **Shape fitting** (2000 iterations): Optimizes SMPL body shape (β parameters) and global scale to match the H1 skeleton proportions in T-pose
2. **Motion fitting** (1000 iterations per motion): Uses differentiable FK + Adadelta optimizer to find H1 joint angles that minimize the positional error between matched robot and SMPL joints

**Key insight**: ASAP required creating a full robot configuration (MuJoCo XML, joint matches, SMPL pose modifiers, virtual extended bodies for hands/head). The H1's 19 DOFs are matched to 14 SMPL joints. Virtual "hand" and "head" links are appended to the kinematic tree for end-effector matching.

**Result**: High-quality retargeting with ankle heights near ground (0.10–0.23m), kick height up to 1.45m, and all joints actively moving.

## Stage 3: Imitation Policy

### Architecture

- **Algorithm**: RSL_RL PPO with [512, 256, 128] MLP, ELU activations
- **Environment**: IsaacLab `DirectRLEnv` with H1 articulation, 4096 parallel envs
- **Observation** (88-dim): current DOFs (19) + velocities (19) + reference DOF targets (19) + root state (13) + key body relative positions (18)
- **Action** (19-dim): residual joint position corrections on top of reference pose (±1.0 rad)

### Reward Design

The reward combines **motion tracking** and **stability** terms:

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| Joint position | 0.35 | `exp(-1.0 * \|\|q - q_ref\|\|²)` | Track reference joint angles |
| Joint velocity | 0.05 | `exp(-0.05 * \|\|q̇ - q̇_ref\|\|²)` | Match motion dynamics |
| Root position | 0.10 | `exp(-5.0 * \|\|p - p_ref\|\|²)` | Track pelvis position |
| Root rotation | 0.05 | `exp(-2.0 * \|\|Δq\|\|²)` | Track pelvis orientation |
| End-effector | 0.20 | `exp(-10.0 * \|\|ee - ee_ref\|\|²)` | Track hands, feet, shoulders |
| **Upright** | **0.15** | `exp(-5.0 * (1 - up_z)²)` | Keep torso vertical |
| **Alive** | **0.10** | `1.0` (constant) | Reward not falling |
| Smoothness | -0.01 | `\|\|a_t - a_{t-1}\|\|²` | Penalize jerky actions |

### Reference State Initialization (RSI)

On reset, each environment is initialized to a random point in the reference motion clip (random joint positions, velocities, root state from the motion data). This acts as a curriculum: the policy learns to track from any phase, not just the start.

## Key Insights & Debugging Lessons

### 1. Root Body Identity Matters
The `reference_body` must be the actual root link of the articulation (pelvis), not a convenient body like torso. Using torso caused RSI to place the pelvis 0.4m too high, making the robot fall on every reset. The policy learned to survive falls instead of imitating motion.

### 2. The Policy Needs to See the Reference
Without reference DOF targets in the observation, the policy must blindly memorize a time-varying trajectory — which is nearly impossible. Adding 19 reference joint angles to the observation gave the policy a clear target, dramatically improving learning speed.

### 3. Residual Actions >> Absolute Actions
Having the policy output full-range joint angles (±π) means that random initial actions send joints to extreme positions, causing violent jittering. Residual actions (small corrections on the reference pose) mean the robot starts close to the reference motion even with an untrained policy.

### 4. Reward Saturation Kills Gradients
With `exp(-scale * error²)`, high scales cause the reward to collapse to 0 for any non-trivial error. The gradient is also ~0, so the policy gets no learning signal. Reducing scales (e.g., joint_pos from 2.0→1.0, end-effector from 40→10) kept the reward in the informative range where gradients are non-zero.

### 5. Tracking Alone ≠ Balance
A kinematically retargeted motion is not guaranteed to be dynamically feasible. The robot could track poses perfectly up to the kick, then fall because the center of mass trajectory isn't physically balanced. Adding explicit stability rewards (upright bonus, alive bonus) and action smoothness gives the policy the freedom to deviate from the reference when balance demands it.

### 6. Body Rotation Data Quality
Placeholder identity quaternions for body rotations made the rotation tracking reward meaningless (penalized any rotation). Computing proper quaternions from ASAP's FK and converting from xyzw→wxyz for IsaacLab's convention was essential.

## Files

| File | Description |
|------|-------------|
| `scripts/visualize_amass.py` | Decode and visualize raw AMASS motion |
| `scripts/retarget_smpl_to_h1.py` | Standalone retargeting script |
| `scripts/visualize_retarget.py` | Visualize retargeted H1 skeleton |
| `outputs/h1_roundhouse.npz` | Retargeted motion (19 DOFs, 20 bodies, 100 frames) |
| `IsaacLab/.../h1_mimic/` | Direct task: env, config, agents, gym registration |
| `train_mimic.sh` | Training launcher (4096 envs, 3000 iters) |
| `eval_mimic.sh` | Policy evaluation with video recording |
