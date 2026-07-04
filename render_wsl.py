import os
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx
import time
import pickle

# Import your environment and network definitions
from train_standing_balance import H1BalanceEnv, ActorCritic

def render_model():
    print("[*] Loading Environment...")
    env = H1BalanceEnv()
    network = ActorCritic(action_dim=env.action_dim)
    
    CHECKPOINT_DIR = "./apex_h1_checkpoints" # Assuming extracted zip is here
    
    # 1. Find latest checkpoint
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"[!] Critical: Cannot find '{CHECKPOINT_DIR}' directory!")
        print("    Please extract your Kaggle zip file here so the path exists.")
        return
        
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("kaggle_ckpt_")]
    if not checkpoint_files:
        print(f"[!] Critical: No checkpoint folders found inside '{CHECKPOINT_DIR}'.")
        return
        
    latest_num = max([int(f.split("_")[-1]) for f in checkpoint_files])
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"kaggle_ckpt_{latest_num}", "params.flax")
    
    print(f"[*] Loading Neural Weights and Welford stats from Step {latest_num}...")
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
        if isinstance(checkpoint_data, tuple) and len(checkpoint_data) == 5:
            params, opt_state, r_mean, r_var, r_count = checkpoint_data
        else:
            print("[!] Critical: Checkpoint format not recognized.")
            return

    # 2. Setup standard MuJoCo CPU structures for the viewer
    m = env.mj_model
    d = mujoco.MjData(m)
    
    # 3. Compile the JAX rendering step
    @jax.jit
    def single_step(p_params, c_state, r_mean, r_var):
        # Retrieve Current Observation
        c_obs = env.get_observation(c_state)
        
        # Apply Welford Normalization exactly as during training
        norm_obs = (c_obs.astype(jnp.float64) - r_mean) / jnp.sqrt(r_var + 1e-8)
        norm_obs = jnp.clip(norm_obs, -10.0, 10.0).astype(jnp.float32)
        
        # In evaluation mode, we take the mean action directly (no noise injection)
        actions, _, _ = network.apply(p_params, norm_obs)
        
        # Step the physics engine forward
        next_state, next_obs, reward, done, timeout, info = env.step(c_state, actions)
        return next_state, actions, done

    # 4. Initialize JAX Physics State
    rng_reset = jax.random.PRNGKey(42)
    state, _ = env.reset(rng_reset)
    
    print("[*] Launching MuJoCo 3D Viewer...")
    print("    Press ESC to exit.")
    
    # 5. Launch the GUI
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Step the GPU/JAX physics simulation
            state, action, done = single_step(params, state, r_mean, r_var)
            
            # Transfer the JAX mjx.Data back to the CPU mujoco.MjData for rendering!
            mjx.get_data(m, d, state.physics)
            
            # Sync the 3D viewer to the new data
            viewer.sync()
            
            # If the robot falls, completely reset the environment
            if done > 0.5:
                rng_reset, _ = jax.random.split(rng_reset)
                state, _ = env.reset(rng_reset)
                print("[!] Robot fell. Resetting simulation...")
                
            # Frame-pacing: Physics control step is 0.05 seconds. 
            # Sleep so the render doesn't run at 1000x speed!
            time_until_next_step = 0.05 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    render_model()
