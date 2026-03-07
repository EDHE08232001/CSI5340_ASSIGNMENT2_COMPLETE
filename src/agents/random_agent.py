"""Random baseline agent."""

import numpy as np

from src.environment import make_env


def run_random_experiment(seed: int, cfg: dict) -> list:
    """Run a random-action agent and return episode rewards."""
    print(f"--- Random | seed={seed} ---")

    rng = np.random.default_rng(seed)
    env = make_env(cfg["env"], seed=seed)

    episode_rewards = []

    for ep in range(cfg["training"]["num_episodes"]):
        obs, _    = env.reset(seed=int(seed + ep))
        ep_reward = 0.0
        done      = False

        while not done:
            action     = int(rng.integers(0, env.n_actions))
            obs, r, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            ep_reward += r

        episode_rewards.append(float(ep_reward))

    env.close()
    return episode_rewards
