import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular replay buffer backed by NumPy arrays."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity  = capacity
        self.state_dim = state_dim
        self._ptr      = 0
        self._size     = 0

        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity,              dtype=np.int32)
        self.rewards     = np.zeros(capacity,              dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity,              dtype=np.float32)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        idx = self._ptr % self.capacity
        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        """Return a random batch as JAX arrays."""
        idxs = rng.integers(0, self._size, size=batch_size)
        return (
            jnp.asarray(self.states[idxs]),
            jnp.asarray(self.actions[idxs]),
            jnp.asarray(self.rewards[idxs]),
            jnp.asarray(self.next_states[idxs]),
            jnp.asarray(self.dones[idxs]),
        )

    def __len__(self) -> int:
        return self._size
