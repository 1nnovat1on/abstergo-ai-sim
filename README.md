# Wind Balance Sim

A lightweight, deterministic 2D balancing toy environment inspired by the project spec. It simulates a platform, vertical stick, and ball under gravity and Perlin-noise wind while training a simple neural policy via multiprocessing.

## Features

- Deterministic physics with fixed timestep (`dt = 1/60`)
- Perlin-driven horizontal wind field affecting the stick and ball
- Discrete platform control actions (-1, 0, +1) with clamped world bounds
- Reward encourages keeping the ball on the stick; episode ends when the ball falls
- Multiprocessing rollout workers with a dependency-free MLP policy trained using REINFORCE

## Usage

Run a short training session with defaults:

```bash
python -m wind_balance_sim
```

Or customize parameters:

```bash
python - <<'PY'
from wind_balance_sim import TrainingConfig, train, EnvConfig

config = TrainingConfig(
    iterations=5,
    steps_per_worker=128,
    num_workers=3,
    envs_per_worker=6,
    env_config=EnvConfig(wind_strength=80.0),
)
train(config)
PY
```

The training loop prints per-iteration progress and exits after the configured number of iterations.
