import os

# [!] KAGGLE MULTI-GPU DEADLOCK FIX [!]
# Kaggle allocates 2x T4 GPUs. JAX deadlocks during cross-PCIe garbage collection on Step 2.
# We completely blindfold JAX to see only 1 GPU before it even boots up!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
jax.config.update("jax_enable_x64", True) # Patch 1: Float64 Precision for Welford
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import mujoco
from mujoco import mjx
import time
import argparse
import sys

# =============================================================================
# GLOBAL TRAINING CONFIGURATION
# =============================================================================
TOTAL_TIMESTEPS = 5

# =============================================================================
# AUTOMATED CROSS-PLATFORM ENVIRONMENT DISCOVERY ENGINE
# =============================================================================
if os.path.exists("/kaggle/input"):
    PLATFORM = "Kaggle"
    XML_SEARCH_ROOT = "/kaggle/input"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints/"
    # Enable JAX Persistent Compilation Cache so restarts skip the 10-minute compile!
    jax.config.update("jax_compilation_cache_dir", "/kaggle/working/jax_cache")
elif os.path.exists("/content"):
    PLATFORM = "Google Colab"
    XML_SEARCH_ROOT = "/content"
    CHECKPOINT_DIR = "/content/checkpoints/apex_h1/"
else:
    PLATFORM = "Local WSL"
    XML_SEARCH_ROOT = "./"
    CHECKPOINT_DIR = "./checkpoints/apex_h1/"

print(f"[*] APEX Bootloader: Detected platform running on [{PLATFORM}]", flush=True)
devices = jax.devices()
print(f"[*] JAX Device Discovery: Using devices {devices}", flush=True)
if devices[0].platform == "cpu":
    print("[!] WARNING: JAX IS RUNNING ON CPU! Ensure Kaggle GPU is activated and Jax is installed correctly.", flush=True)

# Dynamic XML Path Resolver
XML_PATH = None
for root, dirs, files in os.walk(XML_SEARCH_ROOT):
    if "h1.xml" in files:
        XML_PATH = os.path.join(root, "h1.xml")
        break

if XML_PATH is None:
    if PLATFORM == "Kaggle":
        XML_PATH = "/kaggle/input/apex-assets/assets/h1.xml"
    elif PLATFORM == "Google Colab":
        XML_PATH = "/content/assets/h1.xml"
    else:
        XML_PATH = "assets/h1.xml"

print(f"[*] APEX Asset Resolver: Targeting physics model XML at: {XML_PATH}", flush=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
# TITANIUM: MICRO-INERTIA STABILITY PATCH
# =============================================================================
KP_ARRAY = jnp.array([150.0] * 10 + [400.0] * 1 + [10.0] * 8)
KD_ARRAY = jnp.array([10.0] * 10 + [20.0] * 1 + [0.1] * 8)

def quat_rotate_inverse(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    ux = -x
    uy = -y
    uz = -z
    
    cross1_x = uy * v[2] - uz * v[1]
    cross1_y = uz * v[0] - ux * v[2]
    cross1_z = ux * v[1] - uy * v[0]
    
    cross2_x = uy * cross1_z - uz * cross1_y
    cross2_y = uz * cross1_x - ux * cross1_z
    cross2_z = ux * cross1_y - uy * cross1_x
    
    v_rot_x = v[0] + 2.0 * w * cross1_x + 2.0 * cross2_x
    v_rot_y = v[1] + 2.0 * w * cross1_y + 2.0 * cross2_y
    v_rot_z = v[2] + 2.0 * w * cross1_z + 2.0 * cross2_z
    
    return jnp.array([v_rot_x, v_rot_y, v_rot_z])

@flax.struct.dataclass
class EnvState:
    physics: mjx.Data
    rng: jax.Array
    step_counter: jax.Array
    action_history: jax.Array 
    perturbation_timer: jax.Array
    perturbation_force: jax.Array
    survival_steps: jax.Array
    kp_randomization: jax.Array

class H1BalanceEnv:
    def __init__(self, xml_path=XML_PATH):
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            self.mj_model.dof_damping[:] = 0.5
            self.mj_model.dof_armature[:] = 0.05
        except Exception as e:
            print(f"[!] Critical Error loading model from {xml_path}: {e}", flush=True)
            raise e

        self.action_dim = self.mj_model.nu
        self.observation_dim = 45 
        
        for i in range(self.mj_model.ngeom):
            if self.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self.mj_model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE
                
        self.mjx_model = mjx.put_model(self.mj_model)

        qpos_indices = []
        qvel_indices = []
        for i in range(self.mj_model.nu):
            joint_id = self.mj_model.actuator_trnid[i, 0]
            qpos_indices.append(self.mj_model.jnt_qposadr[joint_id])
            qvel_indices.append(self.mj_model.jnt_dofadr[joint_id])

        self.qpos_indices = jnp.array(qpos_indices)
        self.qvel_indices = jnp.array(qvel_indices)

        self.q_nominal = jnp.array([
            0.0, 0.0, -0.4, 0.8, -0.4,
            0.0, 0.0, -0.4, 0.8, -0.4,
            0.0,                      
            0.0, 0.0, 0.0, 0.0,       
            0.0, 0.0, 0.0, 0.0        
        ])

    def reset(self, rng):
        key_init, key_state, key_kp = jax.random.split(rng, 3)
        physics_data = mjx.make_data(self.mjx_model)
        
        qpos = jnp.zeros(self.mjx_model.nq)
        qpos = qpos.at[2].set(0.98) 
        qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0])) 
        qpos = qpos.at[7:7+self.action_dim].set(self.q_nominal)
        
        physics_data = physics_data.replace(qpos=qpos)
        physics_data = mjx.forward(self.mjx_model, physics_data)

        kp_rand = jax.random.uniform(key_kp, shape=(self.action_dim,), minval=0.95, maxval=1.05)

        state = EnvState(
            physics=physics_data,
            rng=key_state,
            step_counter=jnp.array(0, dtype=jnp.int32),
            action_history=jnp.zeros((3, self.action_dim)),
            perturbation_timer=jnp.array(0, dtype=jnp.int32),
            perturbation_force=jnp.zeros(3),
            survival_steps=jnp.array(0, dtype=jnp.int32),
            kp_randomization=kp_rand
        )
        return state, self.get_observation(state)

    def get_observation(self, state):
        d = state.physics
        pelvis_z = d.qpos[2]
        base_quat = d.qpos[3:7]
        base_ang_vel = d.qvel[3:6]
        
        q_current = d.qpos[self.qpos_indices]
        qvel_current = d.qvel[self.qvel_indices]

        projected_gravity = quat_rotate_inverse(base_quat, jnp.array([0.0, 0.0, -1.0]))

        obs = jnp.concatenate([
            jnp.array([pelvis_z]),
            projected_gravity,
            base_ang_vel,
            q_current,
            qvel_current
        ])
        
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0)
        return jnp.clip(obs, -50.0, 50.0)

    def step(self, state, action):
        delayed_action = state.action_history[0]
        new_history = jnp.concatenate([state.action_history[1:], action[None, :]], axis=0)
        q_target = self.q_nominal + jnp.clip(delayed_action, -1.0, 1.0) * jnp.float32(0.20)

        key_trigger, key_force, key_dir, key_next = jax.random.split(state.rng, 4)
        
        trigger_active = jax.random.uniform(key_trigger) < 0.05
        force_mag = jax.random.uniform(key_force, minval=50.0, maxval=150.0)
        angle = jax.random.uniform(key_dir, minval=0.0, maxval=2 * jnp.pi)
        
        new_timer = jnp.where(
            (state.perturbation_timer <= 0) & trigger_active,
            jnp.array(15, dtype=jnp.int32), 
            jnp.maximum(jnp.int32(0), state.perturbation_timer - jnp.int32(1))
        )
        
        new_force = jnp.where(
            (state.perturbation_timer <= 0) & trigger_active,
            jnp.array([jnp.cos(angle) * force_mag, jnp.sin(angle) * force_mag, jnp.float32(0.0)]),
            jnp.where(new_timer > 0, state.perturbation_force, jnp.zeros(3))
        )

        def substep_fn(i, current_physics):
            q_curr = current_physics.qpos[self.qpos_indices]
            qvel_curr = current_physics.qvel[self.qvel_indices]
            
            randomized_kp = KP_ARRAY * state.kp_randomization
            torques = randomized_kp * (q_target - q_curr) - KD_ARRAY * qvel_curr
            torques = jnp.clip(torques, self.mj_model.actuator_ctrlrange[:, 0], self.mj_model.actuator_ctrlrange[:, 1])
            
            next_state = current_physics.replace(ctrl=torques)
            xfrc_applied = next_state.xfrc_applied.at[1, :3].set(new_force)
            next_state = next_state.replace(xfrc_applied=xfrc_applied)
            
            return mjx.step(self.mjx_model, next_state)

        new_physics = jax.lax.fori_loop(0, 10, substep_fn, state.physics)

        pelvis_z = new_physics.qpos[2]
        base_quat = new_physics.qpos[3:7]
        lin_vel_xy = new_physics.qvel[0:2]
        joint_vel_curr = new_physics.qvel[self.qvel_indices]

        # Projected gravity vector for torso tilt checking
        projected_gravity = quat_rotate_inverse(base_quat, jnp.array([0.0, 0.0, -1.0]))

        # 1. Height Reward (Smooth peak at target height, e.g., 0.95 meters)
        target_height = 0.95
        r_height = jnp.exp(-10.0 * jnp.square(pelvis_z - target_height))

        # 2. Upright Posture Reward (Ensure torso points straight up)
        r_posture = jnp.exp(-3.0 * jnp.square(projected_gravity[2] - (-1.0)))

        # 3. Jitter Penalty (Prevent vibrating/twitchy legs)
        r_jitter = -0.001 * jnp.sum(jnp.square(joint_vel_curr))

        # 4. Drift Penalty (Keep from walking away)
        raw_vel_penalty = -0.1 * jnp.sum(jnp.square(lin_vel_xy))
        r_vel_penalty = jnp.clip(raw_vel_penalty, -10.0, 0.0)

        # 5. Energy Penalty
        torques_applied = new_physics.ctrl
        raw_power = -0.005 * jnp.sum(jnp.square(torques_applied))
        r_power = jnp.clip(raw_power, -10.0, 0.0)

        # 6. Action Smoothness Penalty (Motor Protection)
        action_diff = jnp.sum(jnp.square(action - state.action_history[-1]))
        r_action_smoothness = -0.01 * action_diff

        # Total Reward Calculation
        total_reward = (
            1.0 * r_height + 
            1.0 * r_posture + 
            1.0 * r_vel_penalty + 
            1.0 * r_power +
            1.0 * r_jitter +
            1.0 * r_action_smoothness
        )

        # Early Termination (pelvis drops below 0.55m OR tilts more than ~60 deg)
        is_fallen = jnp.logical_or(pelvis_z < 0.55, projected_gravity[2] > -0.5)
        
        # Patch 3: GAE Timeout Bootstrapping
        is_timeout = (state.survival_steps >= 1000)
        timeout = is_timeout.astype(jnp.float32)
        
        needs_reset = jnp.logical_or(is_fallen, is_timeout)
        done = is_fallen.astype(jnp.float32) # done=1 ONLY when it falls over

        total_reward = jnp.where(is_fallen, total_reward - 100.0, total_reward)
        total_reward = jnp.where(jnp.isnan(total_reward), -100.0, total_reward)

        new_survival = state.survival_steps + jnp.int32(1)

        next_state = EnvState(
            physics=new_physics,
            rng=key_next,
            step_counter=state.step_counter + jnp.int32(1),
            action_history=new_history,
            perturbation_timer=new_timer,
            perturbation_force=new_force,
            survival_steps=jnp.where(needs_reset, jnp.int32(0), new_survival),
            kp_randomization=state.kp_randomization
        )

        reset_state, _ = self.reset(key_next)
        
        info = {
            "survival_steps": jnp.where(needs_reset, new_survival, 0)
        }
        
        # Safeguard: cast reset condition to boolean and only apply where to JAX arrays to prevent PyTree TypeError crashes
        reset_bool = needs_reset.astype(bool)
        final_state = jax.tree_util.tree_map(
            lambda x, y: jnp.where(reset_bool, x, y) if isinstance(y, jax.Array) else y,
            reset_state,
            next_state
        )

        return final_state, self.get_observation(final_state), total_reward, done, timeout, info

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array):
        h = nn.Dense(256)(x)
        h = nn.tanh(h)
        h = nn.Dense(256)(h)
        h = nn.tanh(h)
        action_mean = nn.Dense(self.action_dim)(h)
        action_mean = nn.tanh(action_mean) 

        log_std = self.param("log_std", nn.initializers.constant(-1.0), (self.action_dim,))
        log_std = jnp.clip(log_std, -5.0, 1.0)
        
        v = nn.Dense(256)(x)
        v = nn.tanh(v)
        v = nn.Dense(256)(v)
        v = nn.tanh(v)
        value = nn.Dense(1)(v)

        return action_mean, log_std, jnp.squeeze(value, axis=-1)

def run_training():
    print("[*] Initializing parallel JAX H1 balance environments...", flush=True)
    rng = jax.random.PRNGKey(42)
    
    # -------------------------------------------------------------------------
    # PRODUCTION PARAMETERS (PHASE A TRAINING)
    # -------------------------------------------------------------------------
    num_envs = 2
    T_steps = 8
    update_epochs = 2
    num_minibatches = 24
    
    total_batch_size = num_envs * T_steps
    minibatch_size = total_batch_size // num_minibatches

    gamma = 0.99
    lam = 0.90 # Patch 4: GAE Lambda Variance Suppression
    ppo_clip = 0.2
    ent_coef = 0.01

    print(f"\n[🚀] PHASE-A TRAINING ACTIVE: Running {num_envs} envs for {TOTAL_TIMESTEPS} steps. T_steps={T_steps}\n", flush=True)

    # -------------------------------------------------------------------------
    # KAGGLE SAFETY TIMEOUT
    # -------------------------------------------------------------------------
    MAX_RUNTIME_SEC = 500 
    print(f"[*] Kaggle Safety Timeout activated: Script will self-terminate after {MAX_RUNTIME_SEC} seconds.", flush=True)
    
    env = H1BalanceEnv()
    network = ActorCritic(action_dim=env.action_dim)
    
    # Pro-Tier LR Annealing
    lr_schedule = optax.linear_schedule(
        init_value=3e-4, 
        end_value=1e-5, 
        transition_steps=TOTAL_TIMESTEPS * update_epochs * num_minibatches
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )

    rng, key_init = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, env.observation_dim))
    
    def build_initial_network(k_init, d_obs):
        p = flax.core.unfreeze(network.init(k_init, d_obs))
        opt = tx.init(p)
        return p, opt
        
    params, opt_state = build_initial_network(key_init, dummy_obs)
    
    rng_envs = jax.random.split(rng, num_envs)
    rng, loop_rng = jax.random.split(rng)
    # ---------------------------------------------------------
    def build_initial_carry(r_envs, r_loop):
        e_states, obs = jax.vmap(env.reset)(r_envs)
        r_mean = jnp.zeros(env.observation_dim, dtype=jnp.float64) # Patch 1
        r_var = jnp.ones(env.observation_dim, dtype=jnp.float64)
        r_count = jnp.array(1e-4, dtype=jnp.float64)
        return (e_states, obs, r_loop, r_mean, r_var, r_count)
        
    carry_state = build_initial_carry(rng_envs, loop_rng)

    vmap_step = jax.vmap(env.step)
    apply_fn = jax.jit(network.apply)
    
    @jax.jit
    def compute_gae(rew_arr, done_arr, timeout_arr, val_arr, last_vals):
        advantages = jnp.zeros_like(rew_arr)
        last_gae = jnp.zeros(num_envs)
        for t in reversed(range(T_steps)):
            next_value = jnp.where(t == T_steps - 1, last_vals, val_arr[t + 1])
            # Patch 3: GAE Timeout Bootstrapping
            bootstrap_value = jnp.where(timeout_arr[t] > 0.5, next_value, next_value * (1.0 - done_arr[t]))
            delta = rew_arr[t] + gamma * bootstrap_value - val_arr[t]
            
            # Trajectory breaks on ANY reset (fall or timeout)
            traj_broken = jnp.logical_or(done_arr[t] > 0.5, timeout_arr[t] > 0.5)
            non_terminal_for_gae = 1.0 - traj_broken.astype(jnp.float32)
            
            last_gae = delta + gamma * lam * non_terminal_for_gae * last_gae
            advantages = advantages.at[t].set(last_gae)
        return advantages

    @jax.jit
    def generate_trajectory(p_params, current_carry, r_mean, r_var):
        def body_fn(carry, _):
            c_states, c_obses, c_rng, r_mean, r_var, r_count = carry
            
            # Observation Normalization
            norm_obses = (c_obses.astype(jnp.float64) - r_mean) / jnp.sqrt(r_var + 1e-8)
            norm_obses = jnp.clip(norm_obses, -10.0, 10.0).astype(jnp.float32)
            
            means, log_stds, vals = network.apply(p_params, norm_obses)
            stds = jnp.exp(log_stds)
            
            c_rng, key_act = jax.random.split(c_rng)
            noise = jax.random.normal(key_act, means.shape)
            actions = means + stds * noise
            
            log_probs = -0.5 * jnp.sum(
                jnp.square((actions - means) / stds) + 2.0 * log_stds + jnp.log(2 * jnp.pi), axis=-1
            )

            next_states, next_obses, rews, dones, timeouts, infos = vmap_step(c_states, actions)
            
            transition = (c_obses, norm_obses, actions, rews, dones, timeouts, vals, log_probs, infos["survival_steps"])
            return (next_states, next_obses, c_rng, r_mean, r_var, r_count), transition
            
        final_carry, transitions = jax.lax.scan(body_fn, current_carry, None, length=T_steps)
        return final_carry, transitions

    # ---------------------------------------------------------
    # MONOLITHIC JIT TRAINING STEP (PREVENTS ALL RECOMPILATION)
    # ---------------------------------------------------------
    @jax.jit(donate_argnums=(0, 1, 2, 3))
    def train_step(p_params, p_opt_state, c_state, p_rng, update_step):
        # Unpack state including running stats
        c_states, c_obses, c_rng, r_mean, r_var, r_count = c_state
        
        # Patch 2: Entropy Coefficient Decay Schedule
        current_ent_coef = jnp.maximum(1e-5, 0.01 * (1.0 - update_step / TOTAL_TIMESTEPS))
        
        # 1. Rollout
        next_carry_tuple, transitions = generate_trajectory(p_params, c_state, r_mean, r_var)
        next_c_states, next_c_obses, next_c_rng, _, _, _ = next_carry_tuple
        
        raw_obs_arr, norm_obs_arr, act_arr, rew_arr, done_arr, timeout_arr, val_arr, logp_arr, surv_arr = transitions
        
        # 2. GAE
        # Use current running stats to normalize the final observation for GAE
        final_norm_obs = (next_c_obses.astype(jnp.float64) - r_mean) / jnp.sqrt(r_var + 1e-8)
        final_norm_obs = jnp.clip(final_norm_obs, -10.0, 10.0).astype(jnp.float32)
        _, _, last_vals = network.apply(p_params, final_norm_obs)
        advantages = compute_gae(rew_arr, done_arr, timeout_arr, val_arr, last_vals)
        returns = advantages + val_arr
        
        # 3. Welford's Algorithm (Update Running Stats)
        flat_raw_obs = raw_obs_arr.reshape(-1, env.observation_dim).astype(jnp.float64) # Patch 1
        batch_mean = jnp.mean(flat_raw_obs, axis=0)
        batch_var = jnp.var(flat_raw_obs, axis=0)
        batch_count = flat_raw_obs.shape[0]

        delta = batch_mean - r_mean
        tot_count = r_count + batch_count

        new_mean = r_mean + delta * batch_count / tot_count
        m_a = r_var * r_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * r_count * batch_count / tot_count
        new_var = M2 / tot_count
        
        # Assemble next carry state
        next_c_state = (next_c_states, next_c_obses, next_c_rng, new_mean, new_var, tot_count)
        
        # 4. Flatten for PPO
        flat_obs = norm_obs_arr.reshape(-1, env.observation_dim)
        flat_act = act_arr.reshape(-1, env.action_dim)
        flat_val = val_arr.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_logp = logp_arr.reshape(-1)
        flat_adv = advantages.reshape(-1)
        
        # Normalize advantages
        std = jnp.std(flat_adv)
        safe_std = jnp.where(std < 1e-5, 1.0, std)
        normalized_adv = (flat_adv - jnp.mean(flat_adv)) / safe_std
        flat_adv = jnp.where(std < 1e-5, 0.0, normalized_adv)
        
        # 4. PPO Update with Epochs & Mini-batches
        def epoch_step(epoch_carry, _):
            params, opt_state, e_rng = epoch_carry
            e_rng, shuffle_key = jax.random.split(e_rng)
            indices = jax.random.permutation(shuffle_key, total_batch_size)
            
            def minibatch_step(mb_carry, mb_idx):
                mb_params, mb_opt_state = mb_carry
                start_idx = mb_idx * minibatch_size
                
                # Dynamic slice from the shuffled indices
                mb_indices = jax.lax.dynamic_slice(indices, (start_idx,), (minibatch_size,))
                
                mb_obs = flat_obs[mb_indices]
                mb_act = flat_act[mb_indices]
                mb_logp = flat_logp[mb_indices]
                mb_adv = flat_adv[mb_indices]
                mb_ret = flat_ret[mb_indices]
                mb_val = flat_val[mb_indices]
                
                # Pro-Tier Minibatch Advantage Normalization
                mb_adv = (mb_adv - jnp.mean(mb_adv)) / (jnp.std(mb_adv) + 1e-8)
                
                def loss_fn(p):
                    new_means, new_log_stds, new_vals = network.apply(p, mb_obs)
                    new_stds = jnp.exp(new_log_stds)
                    
                    new_logp = -0.5 * jnp.sum(
                        jnp.square((mb_act - new_means) / new_stds) + 2.0 * new_log_stds + jnp.log(2 * jnp.pi), axis=-1
                    )
                    
                    ratio = jnp.exp(new_logp - mb_logp)
                    clip_adv = jnp.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * mb_adv
                    loss_pi = -jnp.mean(jnp.minimum(ratio * mb_adv, clip_adv))
                    
                    # Pro-Tier Value Function Clipping
                    v_clipped = mb_val + jnp.clip(new_vals - mb_val, -ppo_clip, ppo_clip)
                    v_loss_unclipped = jnp.square(new_vals - mb_ret)
                    v_loss_clipped = jnp.square(v_clipped - mb_ret)
                    loss_v = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))
                    
                    total_loss = loss_pi + 0.5 * loss_v - current_ent_coef * jnp.mean(new_log_stds)
                    return total_loss, (loss_pi, loss_v)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (mb_loss, _), grads = grad_fn(mb_params)
                updates, next_mb_opt = tx.update(grads, mb_opt_state, mb_params)
                next_mb_params = optax.apply_updates(mb_params, updates)
                
                return (next_mb_params, next_mb_opt), mb_loss

            (params, opt_state), mb_losses = jax.lax.scan(
                minibatch_step, 
                (params, opt_state), 
                jnp.arange(num_minibatches)
            )
            return (params, opt_state, e_rng), jnp.mean(mb_losses)

        (next_params, next_opt_state, next_rng), epoch_losses = jax.lax.scan(
            epoch_step, 
            (p_params, p_opt_state, p_rng), 
            None, 
            length=update_epochs
        )
        final_loss = jnp.mean(epoch_losses)
        
        # 5. Metrics
        mean_reward = jnp.mean(rew_arr)
        mean_survival = jnp.mean(surv_arr)
        return next_params, next_opt_state, next_c_state, next_rng, final_loss, mean_reward, mean_survival


    start_step = 0
    
    # Checkpoint Discovery Logic (Supports Kaggle Mounted Datasets)
    possible_dirs = [CHECKPOINT_DIR]
    if PLATFORM == "Kaggle":
        possible_dirs.append("/kaggle/input")
    elif PLATFORM == "Google Colab":
        possible_dirs.append("/content")
    else:
        possible_dirs.append("./")
        
    highest_ckpt_path = None
    latest_num = -1
    
    for search_dir in possible_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for d in dirs:
                if d.startswith("kaggle_ckpt_"):
                    try:
                        num = int(d.split("_")[-1])
                        if num > latest_num:
                            # Verify that it contains params.flax
                            if os.path.exists(os.path.join(root, d, "params.flax")):
                                latest_num = num
                                highest_ckpt_path = os.path.join(root, d)
                    except ValueError:
                        pass

    if highest_ckpt_path:
        try:
            print(f"[*] Found existing checkpoint: {highest_ckpt_path}. Restoring state...", flush=True)
            with open(os.path.join(highest_ckpt_path, "params.flax"), "rb") as f:
                import pickle
                checkpoint_data = pickle.load(f)
                if isinstance(checkpoint_data, tuple):
                    if len(checkpoint_data) == 2:
                        params, opt_state = checkpoint_data
                    elif len(checkpoint_data) == 5:
                        params, opt_state, r_mean, r_var, r_count = checkpoint_data
                        # Inject loaded running stats into carry state
                        e_states, obs, r_loop, _, _, _ = carry_state
                        carry_state = (e_states, obs, r_loop, r_mean, r_var, r_count)
                else:
                    params = checkpoint_data # Fallback for old weights-only checkpoints
            start_step = latest_num
        except Exception as e:
            print(f"[!] Warning: Failed to load checkpoint from {highest_ckpt_path}: {e}")
            pass

    print(f"[🚀] Starting Phase-A Training on {num_envs} Parallel Environments...", flush=True)
    comp_start = time.time()
    start_time = time.time()
    
    for update in range(start_step, TOTAL_TIMESTEPS):
        step_start_time = time.time()
        
        if update == start_step:
            print(f"[✓] Initializing JAX Trajectory Kernel compilation (this takes ~10-15 minutes on Kaggle)... Please do not interrupt it!", flush=True)

        params, opt_state, carry_state, rng, loss, mean_reward, mean_survival = train_step(params, opt_state, carry_state, rng, update)
        
        if update == start_step:
            print(f"[✓] JAX Physics Compilation complete! Commencing execution...", flush=True)
        
        # Block until PPO step is fully computed so time measurements are accurate,
        # and to prevent asynchronous GPU memory buildup/deadlock.
        loss.block_until_ready()
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, params)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, opt_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, carry_state)
        
        timeout_triggered = (time.time() - comp_start) > MAX_RUNTIME_SEC
        is_final_step = (update + 1) == TOTAL_TIMESTEPS
        
        elapsed = time.time() - start_time
        avg_time = elapsed / (update - start_step + 1)
        eta_sec = (TOTAL_TIMESTEPS - (update + 1)) * avg_time
        eta_seconds = eta_sec
        step_duration = time.time() - step_start_time
        
        if (update + 1) % 1 == 0:
            print(f" [*] Step {update+1:04d} | Loss: {loss:.4f} | Mean Reward: {mean_reward:.4f} | Mean Survival: {mean_survival:.1f} steps | Step Time: {step_duration:.2f}s | ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s", flush=True)

        if (update + 1) % 20 == 0 or timeout_triggered or is_final_step:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"kaggle_ckpt_{update + 1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            try:
                with open(os.path.join(checkpoint_path, "params.flax"), "wb") as f:
                    import pickle
                    # Extract running stats from carry_state to save
                    _, _, _, r_mean, r_var, r_count = carry_state
                    pickle.dump((params, opt_state, r_mean, r_var, r_count), f)
                print(f"[✓] Checkpoint saved at Step {update + 1} -> {checkpoint_path}", flush=True)
            except Exception as e:
                print(f"[!] Warning: Failed to save checkpoint: {e}")

        if timeout_triggered or is_final_step:
            print(f"\n[!] Safety timeout or final step reached! Zipping checkpoints for easy download...", flush=True)
            try:
                import shutil
                zip_path = os.path.join(XML_SEARCH_ROOT if PLATFORM != "Local WSL" else ".", "apex_h1_checkpoints")
                if PLATFORM == "Kaggle":
                    zip_path = "/kaggle/working/apex_h1_checkpoints"
                shutil.make_archive(zip_path, 'zip', CHECKPOINT_DIR)
                print(f"[✓] Checkpoints zipped successfully to {zip_path}.zip", flush=True)
            except Exception as e:
                print(f"[!] Warning: Failed to zip checkpoints: {e}")
                
            print(f"[!] Exiting gracefully so Kaggle saves the outputs permanently.", flush=True)
            break

    print(f"\n[✓] PHASE-A TRAINING COMPLETE! Checkpoints exported to: {CHECKPOINT_DIR}", flush=True)

if __name__ == "__main__":
    run_training()
