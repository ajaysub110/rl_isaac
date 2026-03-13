# rl_isaac — RSL_RL PPO on Isaac-Humanoid-v0

Train an RSL_RL PPO policy on the `Isaac-Humanoid-v0` environment using
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/), matching the official
benchmark configuration.

## Benchmark Config

| Parameter       | Value   |
|-----------------|---------|
| Environments    | 4096    |
| Rollout steps   | 32      |
| Iterations      | 500     |
| Total steps     | 65.5M   |
| Expected time   | ~198s (RTX 4090) |

## Quick Start

```bash
# 1. Setup (one-time)
chmod +x setup.sh train.sh eval.sh tensorboard.sh
./setup.sh

# 2. Activate environment
conda activate rl_isaac_env

# 3. Train (full benchmark)
./train.sh

# 4. Monitor with TensorBoard
./tensorboard.sh
# Open http://localhost:6006

# 5. Evaluate trained policy
./eval.sh
```

## Project Structure

```
rl_isaac/
├── setup.sh          # Environment & dependency setup
├── train.sh          # Training launcher (benchmark defaults)
├── eval.sh           # Policy evaluation / playback
├── tensorboard.sh    # TensorBoard launcher
├── README.md
├── IsaacLab/         # (cloned during setup) Isaac Lab framework
└── logs/             # (created during training) TensorBoard logs
    └── rsl_rl/
        └── humanoid_direct/
            └── <timestamp>/
                ├── params/       # Saved env & agent YAML configs
                └── model_*.pt    # Checkpoints
```

## Training Options

```bash
# Smoke test (2 iterations)
./train.sh --max_iterations 2

# Custom env count
./train.sh --num_envs 2048

# Resume from checkpoint
./train.sh --resume
```

## Evaluation Options

```bash
# Play latest checkpoint
./eval.sh

# Play a specific run
./eval.sh --load_run 2026-03-12_22-00-00

# Headless evaluation
./eval.sh --headless
```
