import gc
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def one_hot(state: int, n_states: int) -> np.ndarray:
    """Convert an integer state index to a one-hot float32 vector."""
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v


def cleanup() -> None:
    """Run garbage collection and report."""
    gc.collect()
    print("Cleanup done.")
