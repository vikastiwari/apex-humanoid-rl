import jax
import jax.numpy as jnp
import optax
from functools import partial
import mujoco
from mujoco import mjx

from parkour_env import ParkourEnv
from sensor_fusion import SensorFusionActorCritic

@jax.jit
def compute_gae(rewards: jax.Array, values: jax.Array, dones: jax.Array, next_value: jax.Array, gamma: float = 0.99, lmbda: float = 0.95) -> tuple[jax.Array, jax.Array]:
    """
    Computes Generalized Advantage Estimation (GAE) and returns-to-go backwards in time.
    Inputs:
        rewards: shape (T, B)
        values: shape (T, B)
        dones: shape (T, B)
        next_value: shape (B,)
    """
    # Shift values forward to get V(s_{t+1})
    next_values = jnp.concatenate([values[1:], jnp.expand_dims(next_value, 0)], axis=0)
    
    # TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
    deltas = rewards + gamma * next_values * (1.0 - dones) - values
    
    # Backward accumulation of GAE using jax.lax.scan with reverse=True
    def scan_fn(gae_next, xs):
        delta_t, done_t = xs
        gae_t = delta_t + gamma * lmbda * (1.0 - done_t) * gae_next
        return gae_t, gae_t
        
    _, advantages = jax.lax.scan(
        scan_fn, 
        jnp.zeros_like(next_value), 
        (deltas, dones), 
        reverse=True
    )
    
    returns = advantages + values
    return advantages, returns


def ppo_loss(params, rnn_states, obs, actions, behavior_log_probs, advantages, returns, network, clip_eps=0.2, c_val=0.5, c_ent=0.01):
    """
    Computes standard PPO clipped surrogate actor loss, critic MSE value loss, and entropy bonus.
    """
    # 1. Forward pass to get current policy mean, variance parameters, and critic value
    _, action_mean, action_log_std, values = network.apply(params, rnn_states, obs)
    
    # 2. Log probability of continuous actions under diagonal Gaussian distribution
    var = jnp.exp(2.0 * action_log_std)
    log_probs = -0.5 * jnp.sum(((actions - action_mean) ** 2) / var + 2.0 * action_log_std + jnp.log(2.0 * jnp.pi), axis=-1)
    
    # 3. Probability ratio r_t(theta) = pi_theta(a|s) / pi_old(a|s)
    ratios = jnp.exp(log_probs - behavior_log_probs)
    
    # 4. Clipped Actor loss
    surr1 = ratios * advantages
    surr2 = jnp.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # 5. Critic loss (MSE value function loss)
    critic_loss = c_val * jnp.mean(jnp.square(values - returns))
    
    # 6. Policy Entropy (encourage exploration and penalize early collapse of policy variance)
    entropy = jnp.sum(action_log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1)
    entropy_loss = -c_ent * jnp.mean(entropy)
    
    # Total combined loss
    total_loss = actor_loss + critic_loss + entropy_loss
    
    return total_loss, (actor_loss, critic_loss, entropy_loss)


def train_ppo_mjx(num_envs: int = 1024, num_steps: int = 1000, key_seed: int = 42):
    """
    Scaffolding for MJX Vectorized Training Loop using JAX vmap.
    """
    rng = jax.random.PRNGKey(key_seed)
    
    # Create environment
    env = ParkourEnv(base_xml_path="assets/h1.xml", num_obstacles=5)
    action_dim = env.mjx_model.nu
    network = SensorFusionActorCritic(action_dim=action_dim)
    
    # Define optimizer
    optimizer = optax.adam(learning_rate=3e-4)
    
    # Split RNG for initializing parallel environments
    rng, env_rng = jax.random.split(rng)
    env_rngs = jax.random.split(env_rng, num_envs)

    # Vectorize reset and step environments
    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(env.step, in_axes=(0, 0)) # vmap across batch of environment states and action inputs
    
    # Initialize network params using dummy observations
    rng, init_rng = jax.random.split(rng)
    dummy_rnn_state = jnp.zeros((num_envs, 256))
    dummy_obs = env.get_observation(mjx.make_data(env.mjx_model))
    # Batch observations for init
    batched_dummy_obs = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (num_envs, *x.shape)), dummy_obs)
    
    params = network.init(init_rng, dummy_rnn_state, batched_dummy_obs)
    opt_state = optimizer.init(params)
    
    print(f"[*] Scaffold initialized for {num_envs} massive parallel environments via jax.vmap.")

    @jax.jit
    def ppo_update_step(params, opt_state, env_state, rnn_state, rng):
        """
        A single PPO rollout and training update step.
        Compiled End-to-End via JAX JIT for maximum execution speed.
        """
        # 1. ROLLOUT PHASE (collect trajectory buffer on GPU using lax.scan)
        # In a complete rollout, we track states, observations, actions, log_probs, values, rewards, and dones.
        # Here we scaffold the update step assuming we have extracted these trajectories.
        
        # Placeholder shapes for trajectory updates
        T_steps = 128
        total_batch = T_steps * num_envs
        
        # Broadcast dummy observations to the full rollout batch size
        flat_obs = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[0], (total_batch, *x.shape[1:])),
            batched_dummy_obs
        )
        
        flat_rnn_states = jnp.zeros((total_batch, 256))
        batch_actions = jnp.zeros((total_batch, action_dim))
        batch_log_probs = jnp.zeros((total_batch,))
        batch_rewards = jnp.zeros((T_steps, num_envs))
        batch_values = jnp.zeros((T_steps, num_envs))
        batch_dones = jnp.zeros((T_steps, num_envs))
        next_value = jnp.zeros((num_envs,))
        
        # Compute advantages and returns via GAE
        advantages, returns = compute_gae(batch_rewards, batch_values, batch_dones, next_value)
        
        # Flatten time and batch dimensions for learning step
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        
        # Compute PPO loss gradients
        def loss_fn(p):
            loss, _ = ppo_loss(
                p, flat_rnn_states, flat_obs, batch_actions, batch_log_probs, flat_advantages, flat_returns, network
            )
            return loss
            
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        
        # Apply gradients via Optax
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        avg_curriculum = jnp.mean(env_state.curriculum_level)
        
        return new_params, new_opt_state, env_state, rnn_state, rng, loss_val, avg_curriculum

    return ppo_update_step, params, opt_state, rng, env

if __name__ == "__main__":
    import os
    import flax.training.checkpoints as flax_checkpoints
    
    num_envs = 1024 # Scaled up  for GPU
    ppo_update_step, params, opt_state, rng, env = train_ppo_mjx(num_envs=num_envs)
    
    # Initialize physics state & GRU carry state
    rng, loop_rng = jax.random.split(rng)
    # Use vmap reset for parallel environments
    vmap_reset = jax.vmap(env.reset)
    env_rngs = jax.random.split(loop_rng, num_envs)
    env_state, _ = vmap_reset(env_rngs)
    rnn_state = jnp.zeros((num_envs, 256))
    
    checkpoint_dir = os.path.abspath("./checkpoints/apex_h1/")
    
    start_step = 0
    if os.path.exists(checkpoint_dir):
        state_dict = {"params": params, "opt_state": opt_state}
        try:
            latest = flax_checkpoints.latest_step(checkpoint_dir)
            if latest is not None:
                restored = flax_checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state_dict)
                params = restored["params"]
                opt_state = restored["opt_state"]
                start_step = latest
                print(f"[✓] Restored checkpoint from step {start_step}.")
        except Exception as e:
            print(f"[!] Could not restore checkpoint: {e}")
    
    print("[🚀] Launching Phase-B Parkour training loop and compiling update steps...")
    for step in range(start_step + 1, 100001): # Compile and run 100000 training steps
        params, opt_state, env_state, rnn_state, rng, loss, avg_curr = ppo_update_step(
            params, opt_state, env_state, rnn_state, rng
        )
        print(f" [*] Step {step}/100000 completed. Loss: {loss:.4f} [Curriculum Level: {avg_curr:.2f}]")
        
        # Save a checkpoint every 5000 steps
        if step % 5000 == 0:
            flax_checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target={"params": params, "opt_state": opt_state},
                step=step,
                prefix="checkpoint_",
                overwrite=True
            )
            print(f"[✓] Checkpoint successfully saved to {checkpoint_dir}")

