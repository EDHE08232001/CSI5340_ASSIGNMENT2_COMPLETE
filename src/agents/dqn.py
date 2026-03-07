"""
DQN Agent (JAX)

Uses a Q-network (MLP) with experience replay and a periodically synced
target network.  The update step is jit-compiled; the optimizer and gamma
are captured in the closure so that one compiled binary is shared across
all training steps within a single (lr, gamma) experiment.
"""

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import numpy as np

from src.environment import make_env
from src.models import init_mlp_params, mlp_forward, batched_mlp_forward
from src.replay_buffer import ReplayBuffer
from src.utils import one_hot


def make_dqn_update_fn(optimizer, gamma: float):
    """
    Factory: builds and returns a jit-compiled DQN update step.

    Closing over `optimizer` and `gamma` means JAX traces once per
    unique (lr, gamma) combination — acceptable for a hyperparameter sweep.
    """

    def loss_fn(params, target_params, states, actions, rewards, next_states, dones):
        q_vals = batched_mlp_forward(params, states)              # (B, A)
        q_next = batched_mlp_forward(target_params, next_states)  # (B, A)
        # Bellman target — stop_gradient prevents gradients through target net
        q_tgt  = rewards + gamma * jnp.max(q_next, axis=1) * (1.0 - dones)
        q_tgt  = jax.lax.stop_gradient(q_tgt)                    # (B,)
        B      = states.shape[0]
        q_pred = q_vals[jnp.arange(B), actions]                  # (B,)
        return jnp.mean((q_pred - q_tgt) ** 2)

    @jit
    def update_step(params, opt_state, target_params,
                    states, actions, rewards, next_states, dones):
        loss, grads = value_and_grad(loss_fn)(
            params, target_params, states, actions, rewards, next_states, dones
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return update_step


def run_dqn_experiment(seed: int, lr: float, gamma: float, cfg: dict):
    """Train a DQN agent and return (episode_rewards, final_params)."""
    print(f"--- DQN | lr={lr} | gamma={gamma} | seed={seed} ---")

    n_states   = cfg["env"]["grid_size"] ** 2
    n_actions  = 4
    hidden_dim = cfg["dqn"]["hidden_dim"]

    key           = jax.random.PRNGKey(seed)
    params        = init_mlp_params([n_states, hidden_dim, n_actions], key)
    target_params = jax.tree_util.tree_map(jnp.array, params)  # deep copy

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    update_fn = make_dqn_update_fn(optimizer, gamma)

    buffer = ReplayBuffer(cfg["dqn"]["buffer_size"], n_states)
    rng    = np.random.default_rng(seed)
    env    = make_env(cfg["env"], seed=seed)

    eps        = cfg["dqn"]["epsilon_start"]
    eps_end    = cfg["dqn"]["epsilon_end"]
    eps_decay  = cfg["dqn"]["epsilon_decay"]
    batch_size = cfg["dqn"]["batch_size"]
    tgt_freq   = cfg["dqn"]["target_update_freq"]

    episode_rewards = []
    global_step     = 0

    for ep in range(cfg["training"]["num_episodes"]):
        obs, _    = env.reset(seed=int(seed + ep))
        ep_reward = 0.0
        done      = False

        while not done:
            sv = one_hot(obs, n_states)

            if rng.random() < eps:
                action = int(rng.integers(0, n_actions))
            else:
                q_vals = mlp_forward(params, jnp.asarray(sv))
                action = int(jnp.argmax(q_vals))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(sv, action, reward, one_hot(next_obs, n_states), float(done))
            obs          = next_obs
            ep_reward   += reward
            global_step += 1

            if len(buffer) >= batch_size:
                s, a, r, ns, d = buffer.sample(batch_size, rng)
                params, opt_state, _ = update_fn(
                    params, opt_state, target_params, s, a, r, ns, d
                )

            if global_step % tgt_freq == 0:
                target_params = jax.tree_util.tree_map(jnp.array, params)

        eps = max(eps_end, eps * eps_decay)
        episode_rewards.append(float(ep_reward))

    env.close()
    return episode_rewards, params
