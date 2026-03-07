"""
ELG5340 Assignment 2 — Entry Point

Policy Gradient vs Value-Based RL in JAX:
  Compares REINFORCE and DQN on a stochastic 8×8 GridWorld environment
  across a sweep of learning rates, discount factors, and random seeds.

Usage:
    python main.py
"""

import json
import os
import pickle

# Use non-interactive backend for headless servers (clusters, Colab w/ GPU).
# Remove or comment out the next two lines when running in an interactive
# Jupyter environment where you want inline figure display.
import matplotlib
matplotlib.use("Agg")

import yaml

from src.experiments import run_all_experiments
from src.plotting import (
    get_best_key,
    plot_hyperparam_sensitivity,
    plot_individual_runs,
    plot_mean_se,
    plot_policy_heatmap,
)
from src.utils import cleanup, seed_everything


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_dirs(base_dir: str) -> dict:
    dirs = {
        "logs":  os.path.join(base_dir, "logs"),
        "ckpt":  os.path.join(base_dir, "checkpoints"),
        "plots": os.path.join(base_dir, "plots"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    cfg  = load_config(os.path.join(base_dir, "config.yaml"))
    dirs = setup_dirs(base_dir)

    seed_everything(cfg["seed_base"])

    print(f"Base directory : {base_dir}")
    print(f"Seeds          : {cfg['num_seeds']} (base={cfg['seed_base']})")
    print(f"Episodes       : {cfg['training']['num_episodes']}")

    # ── Run all experiments ───────────────────────────────────────────────────
    results = run_all_experiments(cfg, dirs)
    cleanup()

    # ── Save results ──────────────────────────────────────────────────────────
    log_path = os.path.join(dirs["logs"], "results.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved → {log_path}")

    # ── Learning-curve plots ──────────────────────────────────────────────────
    plot_mean_se(results, dirs["plots"])
    plot_individual_runs(results, dirs["plots"])
    plot_hyperparam_sensitivity(results, cfg, dirs["plots"])

    # ── Policy visualisation ──────────────────────────────────────────────────
    for agent in ["dqn", "reinforce"]:
        bk   = get_best_key(results, agent)
        seed = cfg["seed_base"]
        ckpt = os.path.join(dirs["ckpt"], f"{agent}_{bk}_seed{seed}.pkl")
        with open(ckpt, "rb") as f:
            params = pickle.load(f)
        plot_policy_heatmap(params, agent, bk, cfg["env"], seed, dirs["plots"])

    print("\nAll experiments and plots complete.")


if __name__ == "__main__":
    main()
