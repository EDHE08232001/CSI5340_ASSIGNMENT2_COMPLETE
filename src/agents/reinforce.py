"""
REINFORCE Agent (JAX)

On-policy Monte Carlo policy gradient.  Collects a full episode,
computes normalised discounted returns, then updates the policy network
with a single jit-compiled gradient step.
"""

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import numpy as np

from src.environment import make_env
from src.models import init_mlp_params, mlp_forward, batched_mlp_forward
from src.utils import one_hot


def make_reinforce_update_fn(optimizer):
    """Factory: builds and returns a jit-compiled REINFORCE update step."""

    def loss_fn(params, states, actions, returns):
        logits    = batched_mlp_forward(params, states)                          # (T, A)
        log_probs = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)    # log-softmax
        T         = states.shape[0]
        lp_taken  = log_probs[jnp.arange(T), actions]                           # (T,)
        return -jnp.mean(returns * lp_taken)

    @jit
    def update_step(params, opt_state, states, actions, returns):
        loss, grads = value_and_grad(loss_fn)(params, states, actions, returns)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return update_step


def compute_returns(rewards: list, gamma: float) -> np.ndarray:
    """Discounted returns G_t, normalised to zero mean / unit variance."""
    T       = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G       = 0.0
    for t in reversed(range(T)):
        G          = rewards[t] + gamma * G
        returns[t] = G
    std = returns.std()
    if std > 1e-8:
        returns = (returns - returns.mean()) / std
    return returns


def run_reinforce_experiment(seed: int, lr: float, gamma: float, cfg: dict):
    """Train a REINFORCE agent and return (episode_rewards, final_params)."""
    print(f"--- REINFORCE | lr={lr} | gamma={gamma} | seed={seed} ---")

    n_states   = cfg["env"]["grid_size"] ** 2
    n_actions  = 4
    hidden_dim = cfg["reinforce"]["hidden_dim"]

    key       = jax.random.PRNGKey(seed)
    params    = init_mlp_params([n_states, hidden_dim, n_actions], key)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    update_fn = make_reinforce_update_fn(optimizer)

    rng = np.random.default_rng(seed)
    env = make_env(cfg["env"], seed=seed)

    episode_rewards = []

    for ep in range(cfg["training"]["num_episodes"]):
        obs, _ = env.reset(seed=int(seed + ep))

        ep_states  = []
        ep_actions = []
        ep_rewards = []
        done       = False

        while not done:
            sv      = jnp.asarray(one_hot(obs, n_states))
            logits  = mlp_forward(params, sv)
            probs   = np.array(jnp.exp(logits - jax.nn.logsumexp(logits)), dtype=np.float64)
            probs  /= probs.sum()  # renormalise for numerical safety
            action  = int(rng.choice(n_actions, p=probs))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_states.append(one_hot(obs, n_states))
            ep_actions.append(action)
            ep_rewards.append(reward)
            obs = next_obs

        returns = compute_returns(ep_rewards, gamma)

        states_jax  = jnp.asarray(np.array(ep_states,  dtype=np.float32))
        actions_jax = jnp.asarray(np.array(ep_actions, dtype=np.int32))
        returns_jax = jnp.asarray(returns)

        params, opt_state, _ = update_fn(
            params, opt_state, states_jax, actions_jax, returns_jax
        )

        episode_rewards.append(float(sum(ep_rewards)))

    env.close()
    return episode_rewards, params
