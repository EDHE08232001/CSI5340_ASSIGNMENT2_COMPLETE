from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    8x8 stochastic GridWorld for ELG5340 Assignment 2.

    Design notes
    ------------
    - Start: (0, 0)  Goal: (7, 7)
    - Three staggered obstacle rows form a structured maze that rewards
      long-horizon planning and exposes differences between DQN and REINFORCE.
    - Stochastic transitions (noise=0.1): with 10 % probability the intended
      action is replaced by a uniformly random one, modelling real-world
      actuation noise and testing algorithm robustness.
    - Reward shaping: +1.0 goal, -0.01 per step, -0.5 obstacle hit (agent
      bounces back), -0.05 wall collision.  The sparse +1 reward makes credit
      assignment the primary research challenge.
    """

    metadata = {"render_modes": ["ansi"]}

    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3

    _DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

    def __init__(self, cfg_env: dict, seed: Optional[int] = None):
        super().__init__()

        self.grid_size       = cfg_env["grid_size"]
        self.max_steps       = cfg_env["max_steps"]
        self.noise           = cfg_env["noise"]
        self.obstacles       = set(map(tuple, cfg_env["obstacles"]))
        self.goal_reward     = cfg_env["goal_reward"]
        self.step_reward     = cfg_env["step_reward"]
        self.obstacle_reward = cfg_env["obstacle_reward"]
        self.wall_penalty    = cfg_env["wall_penalty"]

        n = self.grid_size
        self.start = (0, 0)
        self.goal  = (n - 1, n - 1)

        self.n_states  = n * n
        self.n_actions = 4

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space      = spaces.Discrete(self.n_actions)

        self._rng   = np.random.default_rng(seed)
        self._steps = 0
        self._pos   = self.start

    def _encode(self, pos: tuple) -> int:
        return pos[0] * self.grid_size + pos[1]

    def _clamp(self, pos: tuple) -> tuple:
        r, c = pos
        return (
            max(0, min(self.grid_size - 1, r)),
            max(0, min(self.grid_size - 1, c)),
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._pos   = self.start
        return self._encode(self._pos), {}

    def step(self, action: int):
        # Stochastic transition: replace action with random one with prob noise
        if self._rng.random() < self.noise:
            action = int(self._rng.integers(0, self.n_actions))

        dr, dc  = self._DELTAS[int(action)]
        new_pos = self._clamp((self._pos[0] + dr, self._pos[1] + dc))

        hit_wall     = (new_pos == self._pos)
        hit_obstacle = (new_pos in self.obstacles)

        if hit_obstacle:
            reward  = self.obstacle_reward
            new_pos = self._pos  # bounce back
        elif new_pos == self.goal:
            reward = self.goal_reward
        else:
            reward = self.step_reward

        if hit_wall and not hit_obstacle:
            reward += self.wall_penalty

        self._pos    = new_pos
        self._steps += 1

        terminated = (self._pos == self.goal)
        truncated  = (self._steps >= self.max_steps)

        return self._encode(self._pos), float(reward), terminated, truncated, {}

    def render(self, mode: str = "ansi") -> str:
        n    = self.grid_size
        grid = [["." for _ in range(n)] for _ in range(n)]
        for (r, c) in self.obstacles:
            grid[r][c] = "X"
        gr, gc = self.goal
        grid[gr][gc] = "G"
        ar, ac = self._pos
        grid[ar][ac] = "A"
        return "\n".join(" ".join(row) for row in grid)


def make_env(cfg_env: dict, seed: int | None = None) -> GridWorldEnv:
    """Convenience factory for GridWorldEnv."""
    return GridWorldEnv(cfg_env, seed=seed)
