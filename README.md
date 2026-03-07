# CSI5340 — Assignment 2: Policy Gradient vs Value-Based RL in JAX

Compares **DQN** (value-based) and **REINFORCE** (policy gradient) against a random baseline on a custom stochastic 8×8 GridWorld, implemented in pure JAX with JIT compilation and `vmap` vectorisation. Experiments were run on the University of Ottawa **Morningstar HPC cluster**.

---

## Table of Contents

- [Environment](#environment)
- [Agents](#agents)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Running on Morningstar (SLURM)](#running-on-morningstar-slurm)
- [Configuration](#configuration)
- [Outputs](#outputs)

---

## Environment

An 8×8 stochastic GridWorld (`src/environment.py`) built on [Gymnasium](https://gymnasium.farama.org/).

- **Start:** `(0, 0)` — **Goal:** `(7, 7)`
- **Actions:** Up, Down, Left, Right
- **Stochastic transitions:** With 10% probability, the intended action is replaced by a uniformly random one
- **Three obstacle rows** forming a structured maze:

```
Row 0:  S . . . . . . .
Row 1:  . . . . . . . .
Row 2:  . X X X X X . .   ← wide horizontal wall
Row 3:  . . . . . . . .
Row 4:  . . . X X X X .   ← right-side barrier
Row 5:  . . . . . . . .
Row 6:  . X X X . . . .   ← left-side barrier
Row 7:  . . . . . . . G
```

| Event            | Reward  |
|------------------|---------|
| Reach goal (7,7) | `+1.0`  |
| Each time step   | `-0.01` |
| Hit obstacle     | `-0.5`  |
| Hit wall         | `-0.05` |

---

## Agents

| Agent       | File                        | Description                                                  |
|-------------|-----------------------------|--------------------------------------------------------------|
| Random      | `src/agents/random_agent.py`| Uniform random action selection. Lower-bound baseline.       |
| DQN         | `src/agents/dqn.py`         | Q-network with experience replay and a target network. JIT-compiled Bellman update via `@jit` and `value_and_grad`. |
| REINFORCE   | `src/agents/reinforce.py`   | On-policy Monte Carlo policy gradient. Full-episode returns normalised to zero mean / unit variance before the JIT-compiled gradient step. |

Both learning agents use a two-layer MLP (`src/models.py`) with a hidden dimension of 64, batched via `vmap`.

---

## Project Structure

```
CSI5340_Assignment2/
├── main.py                 # Entry point — runs sweep, saves results, generates plots
├── config.yaml             # All hyperparameters and environment settings
├── requirements.txt
├── submit.sh               # SLURM submission script for Morningstar
├── env_test.py             # Quick sanity-check for the environment
└── src/
    ├── environment.py      # GridWorldEnv (Gymnasium-compatible)
    ├── models.py           # JAX MLP: init, forward pass, vmap batching
    ├── replay_buffer.py    # Circular replay buffer backed by NumPy (DQN)
    ├── experiments.py      # Hyperparameter sweep runner
    ├── plotting.py         # Learning curves and policy heatmap visualisation
    ├── utils.py            # Seeding, one-hot encoding, cleanup
    └── agents/
        ├── dqn.py          # DQN agent
        ├── reinforce.py    # REINFORCE agent
        └── random_agent.py # Random baseline
```

---

## Setup

Requires Python 3.10+ and a working JAX installation (CPU or GPU).

```bash
# Clone / unzip the project, then:
pip install -r requirements.txt
```

**`requirements.txt`**

```
gymnasium>=0.29
jax>=0.4
jaxlib>=0.4
optax>=0.2
matplotlib>=3.7
numpy>=1.24
pyyaml>=6.0
```

> **GPU note:** The default `jaxlib` install targets CPU. For GPU support, install the CUDA-enabled build per the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) before running.

---

## Running Experiments

```bash
python main.py
```

This runs the full sweep: 3 learning rates × 2 discount factors × 10 seeds for both DQN and REINFORCE, plus the random baseline (500 episodes each). On a GPU the full sweep takes roughly 5 hours.

To do a quick sanity-check of the environment first:

```bash
python env_test.py
```

---

## Running on Morningstar (SLURM)

A ready-to-use SLURM script is provided:

```bash
sbatch submit.sh
```

`submit.sh` requests 1 GPU on the `small` partition with a 24-hour wall time, activates the virtualenv, and runs `main.py`. Stdout and stderr are written to `out/job-<JOBID>.out` and `out/job-<JOBID>.err`.

To set up the environment on Morningstar from scratch:

```bash
# On a login node
module load python/3.10
virtualenv myenv
source myenv/local/bin/activate
pip install -r requirements.txt   # uses the uOttawa pip mirror automatically
```

---

## Configuration

All settings live in `config.yaml`. Key fields:

```yaml
seed_base: 42
num_seeds: 10

env:
  grid_size: 8
  max_steps: 200
  noise: 0.1            # probability of random action override
  goal_reward: 1.0
  step_reward: -0.01
  obstacle_reward: -0.5
  wall_penalty: -0.05

training:
  num_episodes: 500

hyperparameters:
  learning_rates:   [0.001, 0.01, 0.1]
  discount_factors: [0.9, 0.99]

dqn:
  hidden_dim: 64
  buffer_size: 5000
  batch_size: 64
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995
  target_update_freq: 20

reinforce:
  hidden_dim: 64
```

---

## Outputs

All outputs are generated under the project root and are excluded from version control (see `.gitignore`).

| Directory      | Contents                                                                 |
|----------------|--------------------------------------------------------------------------|
| `logs/`        | `results.json` — all episode rewards for every agent, config, and seed  |
| `checkpoints/` | Trained parameters as `.pkl` files, named `<agent>_lr<lr>_gamma<g>_seed<s>.pkl` |
| `plots/`       | `plot1_mean_se.png` — mean ± SE learning curves<br>`plot2_individual_<agent>.png` — all 10 seed runs overlaid<br>`plot3_hyperparam_sensitivity.png` — sensitivity across all configs<br>`policy_<agent>_<config>_seed<s>.png` — greedy policy heatmaps with action arrows |
| `out/`         | SLURM job stdout/stderr logs                                             |
