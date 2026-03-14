# rl_isaac — Progress Log

## ✅ Environment Setup (2026-03-12)

- Created conda env `rl_isaac_env` (Python 3.11) via `/opt/miniforge3/bin/conda`
- Installed **Isaac Sim 5.1.0** (pip, ~6GB including extscache)
- Installed **PyTorch 2.7.0+cu128** (CUDA-enabled, RTX 4090 confirmed)
- Cloned **Isaac Lab** → `/workspace/rl_isaac/IsaacLab/`
- Ran `isaaclab.sh --install rsl_rl` — installed IsaacLab extensions + RSL_RL + TensorBoard

## ✅ Project Files Created (2026-03-12)

| File | Purpose |
|------|---------|
| `setup.sh` | One-time environment setup (conda + Isaac Sim + Isaac Lab) |
| `train.sh` | Training launcher — benchmark defaults (4096 envs, 500 iters) |
| `eval.sh` | Policy evaluation / playback |
| `tensorboard.sh` | TensorBoard log viewer |
| `README.md` | Project docs |

## ✅ Verification (2026-03-12)

- **Imports OK**: PyTorch 2.7.0+cu128, CUDA available, RTX 4090 detected, RSL_RL, IsaacLab
- **Smoke test (2 iters)**: Completed in 7.21s, 80k steps/sec, logs written to `logs/rsl_rl/humanoid/`
- **TensorBoard**: Event files confirmed at `logs/rsl_rl/humanoid/<timestamp>/`

## ✅ Stage 1: Visualize AMASS Motion (2026-03-13)

- `scripts/visualize_amass.py`: decoded roundhouse kick NPZ (100 frames @ 30fps)
- Generated `outputs/amass_roundhouse.gif` confirming correct motion

## ✅ Stage 2: SMPL → H1 Retargeting via ASAP (2026-03-13)

- Set up [ASAP](https://github.com/LeCAR-Lab/ASAP) pipeline with all dependencies (`smpl_sim`, `chumpy`, `mujoco`)
- Created H1 19-DOF YAML config: joint matches, SMPL pose modifiers, extended bodies (hands/head)
- **Shape fitting** (loss=50.96) → `shape_optimized_v1.pkl`
- **Motion retargeting** (loss=58.59) → ankle Z [0.1–0.23m], kick Z up to 1.45m
- Output: `outputs/h1_roundhouse.npz` (100 frames, 19 DOFs, 20 bodies, proper wxyz rotations)

## ✅ Stage 3: Imitation Policy Training (2026-03-13–14)

- Created `Isaac-H1-Mimic-Direct-v0` direct task in IsaacLab
- **DeepMimic-style reward**: weighted exp() tracking (joint pos/vel, root pos/rot, end-effector)
- **Stability rewards**: upright bonus, alive bonus, action smoothness penalty
- **Residual action space**: policy outputs corrections on reference pose (±1.0 rad)
- **Reference State Initialization (RSI)**: reset to random points in motion clip
- Trained with RSL_RL PPO (4096 envs, [512,256,128] MLP, 3000 iterations)
- **Result**: Robot successfully performs roundhouse kick and recovers balance

### Key Bug Fixes
1. `reference_body`: `torso_link` → `pelvis` (root link mismatch caused +0.4m spawn offset)
2. Body rotations: identity placeholders → proper wxyz from ASAP FK
3. Observation: added reference DOF targets (obs 69→88) so policy knows the tracking goal
4. Action space: full-range absolute → residual on reference (less jitter, easier learning)
5. Reward scales: reduced to prevent exp() saturation (zero gradient problem)
6. Balance rewards: upright + alive + smoothness to prevent post-kick falls
