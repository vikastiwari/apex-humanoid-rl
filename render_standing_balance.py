import os
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import checkpoints
import mujoco
import mujoco.viewer
import time
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
XML_PATH = "assets/h1.xml"
CHECKPOINT_DIR = "./checkpoints/" 

KP_ARRAY = np.array([150.0] * 10 + [400.0] * 1 + [10.0] * 8)
KD_ARRAY = np.array([10.0] * 10 + [20.0] * 1 + [0.1] * 8)

# ==============================================================================
# PPO ACTOR-CRITIC NETWORK
# ==============================================================================
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
        value = jnp.squeeze(value, axis=-1)
        
        return action_mean, log_std, value

# ==============================================================================
# OBSERVATION EXTRACTION
# ==============================================================================
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
    
    return np.array([v_rot_x, v_rot_y, v_rot_z])

def get_observation(model, data, action_dim=19):
    pelvis_z = np.array([data.qpos[2]])
    base_quat = data.qpos[3:7]
    joint_pos = data.qpos[7:7+action_dim]
    base_ang_vel = data.qvel[3:6]
    joint_vel = data.qvel[6:6+action_dim]
    
    projected_gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
    
    obs = np.concatenate([pelvis_z, projected_gravity, base_ang_vel, joint_pos, joint_vel])
    return jnp.array(obs, dtype=jnp.float32)

# ==============================================================================
# QUATERNION TO EULER
# ==============================================================================
def quat_to_euler(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
    return roll, pitch

# ==============================================================================
# MAIN INFERENCE LOOP
# ==============================================================================
def main():
    print(f"[*] JAX Device Discovery: Using devices {jax.devices()}")
    if not os.path.exists(XML_PATH):
        print(f"[!] Cannot find {XML_PATH}. Ensure assets exist.")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    
    # Convert cylinders to capsules for consistency with training
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE
            
    model.dof_damping[:] = 0.5
    model.dof_armature[:] = 0.05
    
    # Aesthetic Upgrade: Light-Theme H1 and Scene Environment
    for i in range(model.ngeom):
        # High-Contrast Floor Remap
        if i == 0 or model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            # Re-establish ground plane by NOT stripping matid, just recolor
            model.geom_rgba[i] = [0.75, 0.75, 0.75, 1.0]
        else:
            # Break Material Associations to force RGBA
            model.geom_matid[i] = -1
            
            # If default color is dark/black (mean RGB < 0.2)
            if np.mean(model.geom_rgba[i][:3]) < 0.2:
                model.geom_rgba[i] = [0.85, 0.85, 0.85, 1.0]
            
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in ["pelvis", "left_ankle_link", "right_ankle_link"]:
                model.geom_rgba[i] = [1.0, 0.45, 0.0, 1.0] # APEX Orange
                
    # Scene Embedded Light Boost
    if model.nlight > 0:
        for l in range(model.nlight):
            model.light_diffuse[l] = [0.9, 0.9, 0.9]
            model.light_specular[l] = [0.2, 0.2, 0.2]
            
    # Force MuJoCo to render a visible grid, a clean FOV, and a light studio gray background
    try:
        model.vis.rgba.grid[:] = [0.5, 0.5, 0.5, 1.0]
        model.vis.rgba.background[:] = [0.2, 0.2, 0.2, 1.0]
        model.vis.rgba.background2[:] = [0.2, 0.2, 0.2, 1.0] # Remove gradient
        model.vis.global_.fovy = 60.0
    except AttributeError:
        pass
        
    try:
        model.vis.global_.ambient[0] = 0.4
        model.vis.global_.ambient[1] = 0.4
        model.vis.global_.ambient[2] = 0.4
    except AttributeError:
        pass # Silently fallback if viewer bindings do not expose this property directly
        
    data = mujoco.MjData(model)
    
    action_dim = model.nu # 19
    q_nominal = np.array([
        0.0, 0.0, -0.4, 0.8, -0.4,
        0.0, 0.0, -0.4, 0.8, -0.4,
        0.0,                      
        0.0, 0.0, 0.0, 0.0,       
        0.0, 0.0, 0.0, 0.0        
    ])
    
    # Initialize Network
    network = ActorCritic(action_dim=action_dim)
    dummy_obs = jnp.zeros((1, 1 + 3 + 3 + action_dim + action_dim))
    
    rng = jax.random.PRNGKey(0)
    params = network.init(rng, dummy_obs)
    
    print(f"[*] Looking for checkpoint at {CHECKPOINT_DIR}...")
    r_mean = jnp.zeros(1 + 3 + 3 + action_dim + action_dim)
    r_var = jnp.ones(1 + 3 + 3 + action_dim + action_dim)
    try:
        import pickle
        possible_dirs = [CHECKPOINT_DIR, "./"]
        loaded = False
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
                                if os.path.exists(os.path.join(root, d, "params.flax")):
                                    latest_num = num
                                    highest_ckpt_path = os.path.join(root, d)
                        except ValueError:
                            pass
        
        if highest_ckpt_path:
            ckpt_file = os.path.join(highest_ckpt_path, "params.flax")
            if os.path.exists(ckpt_file):
                    with open(ckpt_file, "rb") as f:
                        ckpt_state = pickle.load(f)
                    
                    # Unpack the new checkpoint format if it is a tuple
                    if isinstance(ckpt_state, tuple):
                        if len(ckpt_state) == 2:
                            weights_only, _ = ckpt_state
                        elif len(ckpt_state) == 5:
                            weights_only, _, r_mean_val, r_var_val, _ = ckpt_state
                            r_mean = jnp.array(r_mean_val)
                            r_var = jnp.array(r_var_val)
                            print("[*] Loaded running stats (Welford) from checkpoint.")
                        else:
                            weights_only = ckpt_state[0]
                    else:
                        weights_only = ckpt_state

                    # Ensure it has the correct outer dict structure for flax network.apply
                    if isinstance(weights_only, dict) and 'params' in weights_only:
                        params_to_load = weights_only
                    else:
                        params_to_load = {'params': weights_only}
                        
                    try:
                        # Verify that the loaded weights match the new observation space (45-dim)
                        network.apply(params_to_load, dummy_obs)
                        
                        params = params_to_load
                        
                        # Verify weights to show real weights are loaded
                        weight_sum = 0.0
                        weight_count = 0
                        if 'params' in params:
                            for layer_name, layer_params in params['params'].items():
                                if isinstance(layer_params, dict):
                                    for param_name, param_val in layer_params.items():
                                        weight_sum += np.sum(np.abs(param_val))
                                        weight_count += param_val.size
                                elif isinstance(layer_params, (np.ndarray, jax.Array)):
                                    weight_sum += np.sum(np.abs(layer_params))
                                    weight_count += layer_params.size
                        mean_abs_weight = weight_sum / max(1, weight_count)
                        
                        print(f"[✓] Successfully loaded trained parameters from pickle checkpoint: {ckpt_file}")
                        print(f"[✓] WEIGHT VERIFICATION: Loaded {weight_count} weight values with Mean Absolute Magnitude = {mean_abs_weight:.6f}")
                        loaded = True
                    except Exception as e:
                        print(f"[!] Architecture mismatch detected: {e}")
                        print("[!] The observation space has changed (old weights incompatible). Defaulting to RANDOM weights.")
                        # Do not set loaded = True, so we fallback to random params.
        if not loaded:
            print("[!] No valid checkpoint folder ('kaggle_ckpt_*') with 'params.flax' found. Defaulting to RANDOM weights.")
    except Exception as e:
        print(f"[!] Checkpoint deserialization error: {e}")
        print("[!] No valid checkpoint loaded. Defaulting to RANDOM weights for physical dry-run testing.")

    @jax.jit
    def get_action(p, obs, mean, var):
        norm_obs = (obs.astype(jnp.float64) - mean) / jnp.sqrt(var + 1e-8)
        norm_obs = jnp.clip(norm_obs, -10.0, 10.0).astype(jnp.float32)
        action_mean, _, _ = network.apply(p, norm_obs)
        return action_mean

    action_history = np.zeros((3, action_dim))
    current_action = np.zeros(action_dim)
    
    perturbation_active = "None"
    perturbation_timer = 0
    perturbation_force = np.zeros(2)

    def key_callback(keycode):
        nonlocal perturbation_active, perturbation_timer, perturbation_force
        try:
            key = chr(keycode).lower()
            if key == 'l':
                perturbation_active = "Lateral Push"
                perturbation_timer = int(0.2 / model.opt.timestep)
                perturbation_force = np.array([0.0, 120.0]) # Y-axis force
                print("\n[TEST RIG] Lateral Push Applied! (120N)")
            elif key == 's':
                perturbation_active = "Sagittal Push"
                perturbation_timer = int(0.2 / model.opt.timestep)
                perturbation_force = np.array([120.0, 0.0]) # X-axis force
                print("\n[TEST RIG] Sagittal Push Applied! (120N)")
        except ValueError:
            pass

    print("\n==================================================")
    print("    [🚀] REAL-TIME TRAINED H1 INFERENCE TELEMETRY")
    print("    [CONTROLS] 'L': Lateral Push, 'S': Sagittal")
    print("==================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 135.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
        
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.98
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:7+action_dim] = q_nominal
        mujoco.mj_forward(model, data)
        last_reset_time = data.time
        last_log_time = data.time
        
        step = 0
        dt = model.opt.timestep
        control_decimation = int(0.02 / dt) # 4 steps if dt=0.005
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- 1. CONTROL LOOP (50Hz) ---
            if step % control_decimation == 0:
                obs = get_observation(model, data, action_dim)
                obs_batch = jnp.expand_dims(obs, axis=0)
                
                # Inference
                act_jax = get_action(params, obs_batch, r_mean, r_var)
                act_np = np.array(act_jax[0])
                if step < 500:
                    roll, pitch = quat_to_euler(data.qpos[3:7])
                    print(f"[DEBUG] Step {step:03d} | Pelvis Z: {data.qpos[2]:.3f}m | Roll: {np.degrees(roll):5.1f}° | Pitch: {np.degrees(pitch):5.1f}° | action mean: {np.mean(np.abs(act_np)):.4f} | range: [{np.min(act_np):.2f}, {np.max(act_np):.2f}]", flush=True)

                
                # Latency Queue
                delayed_action = action_history[0]
                action_history = np.concatenate([action_history[1:], act_np[np.newaxis, :]], axis=0)
                
                current_action = delayed_action * 0.20
                
            # --- 2. PHYSICS LOOP (200Hz) ---
            q = data.qpos[7:7+action_dim]
            q_vel = data.qvel[6:6+action_dim]
            
            q_target = q_nominal + current_action
            torques = KP_ARRAY * (q_target - q) - KD_ARRAY * q_vel
            torques = np.clip(torques, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            
            data.ctrl[:] = torques
            
            # Apply Perturbation
            if perturbation_timer > 0:
                data.xfrc_applied[1, 0] = perturbation_force[0]
                data.xfrc_applied[1, 1] = perturbation_force[1]
                perturbation_timer -= 1
                if perturbation_timer <= 0:
                    data.xfrc_applied[1, :2] = 0.0
                    perturbation_active = "None"
            
            mujoco.mj_step(model, data)
            
            # --- VISUAL CONTACT HIGHLIGHTING ---
            floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            contact_active = False
            
            for i in range(data.ncon):
                contact = data.contact[i]
                if contact.geom1 == floor_geom_id or contact.geom2 == floor_geom_id:
                    contact_active = True
                    break
                    
            if floor_geom_id != -1:
                if contact_active:
                    # Light Green glow on touch
                    model.geom_rgba[floor_geom_id] = [0.2, 0.8, 0.2, 1.0] 
                else:
                    # Default Cream-gray
                    model.geom_rgba[floor_geom_id] = [0.75, 0.75, 0.75, 1.0] 
            
            # Ground-Fall Safety Monitor
            if data.qpos[2] < 0.55:
                duration = data.time - last_reset_time
                print(f"\n[LOG] Fall detected! Humanoid stood for {duration:.2f} simulated seconds. Resetting simulation to standing...", flush=True)
                mujoco.mj_resetData(model, data)
                data.qpos[2] = 0.98
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                data.qpos[7:7+action_dim] = q_nominal
                mujoco.mj_forward(model, data)
                last_reset_time = data.time
                last_log_time = data.time
                perturbation_timer = 0
                perturbation_active = "None"
                data.xfrc_applied[1, :2] = 0.0
                
            viewer.sync()
            
            # --- 3. TELEMETRY (10Hz) ---
            # if step % int(0.1 / dt) == 0:
            #     z = data.qpos[2]
            #     roll, pitch = quat_to_euler(data.qpos[3:7])
            #     power = np.sum(np.abs(torques * q_vel))
            #     
            #     # ANSI \033[H moves to top-left. We add \033[J to clear the screen below it to prevent trailing characters.
            #     # However, to avoid clearing previous initialization logs, we can just move cursor up.
            #     # But the prompt suggested \033[H or similar. A safe full-clear for a dashboard is \033[H\033[2J.
            #     # Let's use \033[H\033[0J to reset from top.
            #     dashboard = (
            #         "\033[H\033[0J"
            #         "==================================================================\n"
            #         "                 APEX PROJECT: HUMANOID INFRASTRUCTURE            \n"
            #         "==================================================================\n"
            #         f"[STATE MONITOR]  Pelvis Z: {z:.3f}m  |  Roll: {np.degrees(roll):5.1f}°  |  Pitch: {np.degrees(pitch):5.1f}°\n"
            #         f"[KINETIC DRAW]   Joint Power: {power:6.1f} W  |  Substeps: 10 (500Hz)\n"
            #         f"[TEST RIG ENG]   Active Push Force: {perturbation_active}\n"
            #         "=================================================================="
            #     )
            #     print(dashboard, flush=True)

            # Continuous standing logging (every 2.0 simulated seconds)
            if data.time - last_log_time >= 2.0:
                current_duration = data.time - last_reset_time
                z = data.qpos[2]
                roll, pitch = quat_to_euler(data.qpos[3:7])
                print(f"[LOG] Humanoid standing. Duration: {current_duration:.1f}s | Pelvis Z: {z:.3f}m | Roll: {np.degrees(roll):.1f}° | Pitch: {np.degrees(pitch):.1f}°", flush=True)
                last_log_time = data.time
            
            step += 1
            
            # Real-time sync
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
