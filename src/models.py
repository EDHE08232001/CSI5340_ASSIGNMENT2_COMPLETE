from typing import List

import jax
import jax.numpy as jnp
from jax import vmap


def random_layer_params(m: int, n: int, key, scale: float = 1e-2):
    """Return (W, b) initialised from a zero-mean Gaussian."""
    w_key, b_key = jax.random.split(key)
    return (
        scale * jax.random.normal(w_key, (n, m)),
        scale * jax.random.normal(b_key, (n,)),
    )


def init_mlp_params(sizes: List[int], key):
    """Initialise parameters for a fully-connected MLP with the given layer sizes."""
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        random_layer_params(m, n, k)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def relu(x):
    return jnp.maximum(0.0, x)


def mlp_forward(params, x):
    """Single-sample MLP forward pass. Returns raw logits."""
    activations = x
    for w, b in params[:-1]:
        activations = relu(jnp.dot(w, activations) + b)
    w, b = params[-1]
    return jnp.dot(w, activations) + b


# Vectorised version over the batch dimension (params are shared)
batched_mlp_forward = vmap(mlp_forward, in_axes=(None, 0))
