"""
Experiment runner: sweeps over learning rates, discount factors, and seeds
for DQN, REINFORCE, and the random baseline.
"""

import os
import time
import pickle

import numpy as np

from src.agents.dqn import run_dqn_experiment
from src.agents.reinforce import run_reinforce_experiment
from src.agents.random_agent import run_random_experiment


def run_all_experiments(cfg: dict, dirs: dict) -> dict:
    """
    Run all experiments and return a nested results dictionary.

    Structure
    ---------
    {
        "random":    {"default": [{"seed": ..., "episode_rewards": [...]}, ...]},
        "dqn":       {"lr<lr>_gamma<g>": [...], ...},
        "reinforce": {"lr<lr>_gamma<g>": [...], ...},
    }
    """
    lrs    = cfg["hyperparameters"]["learning_rates"]
    gammas = cfg["hyperparameters"]["discount_factors"]
    seeds  = [cfg["seed_base"] + i for i in range(cfg["num_seeds"])]

    results = {"random": {}, "dqn": {}, "reinforce": {}}

    # ── Random Baseline ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Running Random Baseline")
    print("=" * 60)
    results["random"]["default"] = []
    for seed in seeds:
        rewards = run_random_experiment(seed, cfg)
        results["random"]["default"].append({"seed": seed, "episode_rewards": rewards})
        print(f"  seed={seed} | mean_reward={np.mean(rewards):.4f}")

    # ── DQN ───────────────────────────────────────────────────────────────────
    for lr in lrs:
        for gamma in gammas:
            hp_key = f"lr{lr}_gamma{gamma}"
            print("=" * 60)
            print(f"Running DQN | lr={lr} | gamma={gamma}")
            print("=" * 60)
            results["dqn"][hp_key] = []

            for seed in seeds:
                t0 = time.time()
                rewards, params = run_dqn_experiment(seed, lr, gamma, cfg)
                elapsed = time.time() - t0

                results["dqn"][hp_key].append({
                    "seed": seed, "lr": lr, "gamma": gamma,
                    "episode_rewards": rewards, "elapsed_s": round(elapsed, 2),
                })

                ckpt_path = os.path.join(dirs["ckpt"], f"dqn_{hp_key}_seed{seed}.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump(params, f)

                print(f"  seed={seed} | mean_reward={np.mean(rewards):.4f} | time={elapsed:.1f}s")

    # ── REINFORCE ─────────────────────────────────────────────────────────────
    for lr in lrs:
        for gamma in gammas:
            hp_key = f"lr{lr}_gamma{gamma}"
            print("=" * 60)
            print(f"Running REINFORCE | lr={lr} | gamma={gamma}")
            print("=" * 60)
            results["reinforce"][hp_key] = []

            for seed in seeds:
                t0 = time.time()
                rewards, params = run_reinforce_experiment(seed, lr, gamma, cfg)
                elapsed = time.time() - t0

                results["reinforce"][hp_key].append({
                    "seed": seed, "lr": lr, "gamma": gamma,
                    "episode_rewards": rewards, "elapsed_s": round(elapsed, 2),
                })

                ckpt_path = os.path.join(dirs["ckpt"], f"reinforce_{hp_key}_seed{seed}.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump(params, f)

                print(f"  seed={seed} | mean_reward={np.mean(rewards):.4f} | time={elapsed:.1f}s")

    return results
