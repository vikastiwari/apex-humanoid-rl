import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
from flax.training import checkpoints

# Force JAX to run on CPU for local WSL visualizer rendering
os.environ["JAX_PLATFORMS"] = "cpu"

print("=" * 60)
print("     PROJECT APEX - UNBREAKABLE VERIFICATION VISUALIZER")
print("=" * 60)

# ==========================================
# 1. PATH CONFIGURATION & IMPORTS
# ==========================================
sys.path.append(os.path.abspath("."))
try:
    from parkour_env import ParkourEnv
    from train_mjx import SensorFusionActorCritic
    print("[*] Environment and Network loaded successfully.")
except ImportError as e:
    print(f"[!] Critical Import Error: {e}")
    sys.exit(1)

# ==========================================
# 2. ENVIRONMENT & PROCEDURAL MODEL
# ==========================================
print("[*] Preparing physics pipeline and procedural obstacles...")
env = ParkourEnv(base_xml_path="assets/h1.xml")
model = env.mj_model 
data = mujoco.MjData(model)

# ==========================================
# 3. BULLETPROOF ROBOT IDENTIFICATION
# ==========================================
# Mathematically locate the robot's root body by finding the only Free Joint!
robot_free_jnt_id = -1
for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        robot_free_jnt_id = i
        break

if robot_free_jnt_id == -1:
    print("[!] ERROR: Could not find a free joint! Is the humanoid loaded?")
    sys.exit(1)

# Extract exact memory addresses for the robot's root
robot_body_id = model.jnt_bodyid[robot_free_jnt_id]
robot_qpos_adr = model.jnt_qposadr[robot_free_jnt_id]

print(f" [*] Found Robot Root Body ID: {robot_body_id}")

# Find platform sliders
platform_joint_ids = []
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if joint_name and "platform" in joint_name:
        platform_joint_ids.append(i)

# ==========================================
# 4. LOAD TRAINED PARAMETERS
# ==========================================
checkpoint_dir = os.path.abspath("./checkpoints/apex_h1/")
print(f"[*] Loading weights from: {checkpoint_dir}")

from mujoco import mjx
mjx_data = mjx.put_data(model, data)
raw_obs = env.get_observation(mjx_data)
batched_dummy_obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), raw_obs)

dummy_rnn = jnp.zeros((1, 256))
key = jax.random.PRNGKey(0)

# Recreate policy network matching Phase B configurations (using model.nu for actions)
network = SensorFusionActorCritic(action_dim=model.nu)
initial_variables = network.init(key, dummy_rnn, batched_dummy_obs)

try:
    # CRITICAL RESTORE FIX: Use target=initial_variables["params"] to match saved shape perfectly!
    restored_params = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=initial_variables["params"])
    if restored_params is not None:
        params = restored_params
        print("[✓] Successfully loaded Phase B trained parameters!")
    else:
        raise FileNotFoundError("Checkpoints restore returned None.")
except Exception as e:
    print(f"[!] Failed to load checkpoint: {str(e)}")
    print("[!] Fallback: Running visualizer with random parameters.")
    params = initial_variables["params"]

@jax.jit
def policy_step(params, rnn_state, obs):
    outputs = network.apply({"params": params}, rnn_state, obs)
    
    # BULLETPROOF SHAPE-HUNTING UNPACKER
    # Safely unpack regardless of what order the network returns items (RNN first, or Action first)
    out_list = outputs if isinstance(outputs, (tuple, list)) else [outputs]
    
    # Default fallbacks
    action = jnp.zeros((model.nu,))
    new_rnn = rnn_state
    action_found = False
    
    for item in out_list:
        # 1. Check if the item is a TFP Distribution object
        if hasattr(item, 'mean') and hasattr(item, 'variance'):
            action = item.mean()
            action_found = True
        elif hasattr(item, 'sample') and not hasattr(item, 'mean'):
            action = item.mode() if hasattr(item, 'mode') else item.sample(seed=jax.random.PRNGKey(0))
            action_found = True
        # 2. Check if the item is a raw JAX array and match by shape!
        elif hasattr(item, 'shape'):
            # Catch the Memory State (256 dimensions)
            if item.shape == (256,) or item.shape == (1, 256):
                new_rnn = item
            # Catch the Motor Action (Matches model.nu, usually 19)
            elif (item.shape == (model.nu,) or item.shape == (1, model.nu)) and not action_found:
                action = item
                action_found = True
                
    return action, new_rnn

# ==========================================
# 5. UNBREAKABLE RESET FUNCTION
# ==========================================
def reset_simulation_state():
    """Resets the robot securely using the calculated memory addresses."""
    global rnn_state, action, success_streak, last_robot_x, total_ticks
    
    mujoco.mj_resetData(model, data)
    
    # Safely lift the exact free joint up to 1.05m
    data.qpos[robot_qpos_adr + 2] = 1.05
    data.qpos[robot_qpos_adr + 3:robot_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    
    # Calculate initial kinematics
    mujoco.mj_forward(model, data)
    
    rnn_state = jnp.zeros((1, 256))
    action = jnp.zeros((model.nu,))
    success_streak = 0
    total_ticks = 0
    last_robot_x = data.xpos[robot_body_id][0]

# Initialize state
reset_simulation_state()

# ==========================================
# 6. EXECUTION LOOP WITH REAL-TIME TELEMETRY
# ==========================================
print("[🚀] Launching Real-time 3D Rendering & Telemetry Dashboard...")

control_decimation = 20
step_counter = 0
curriculum_level = 0.0
total_ticks = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = robot_body_id
    viewer.cam.distance = 4.5
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 90

    # Draw contact forces (Purple/Green vectors on the floor)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    print("\n" + "="*50)
    print("         [🚀] REAL-TIME APEX TELEMETRY DASHBOARD")
    print("="*50)

    while viewer.is_running():
        step_start = time.time()

        if step_counter % control_decimation == 0:
            mjx_data = mjx.put_data(model, data)
            raw_obs = env.get_observation(mjx_data)
            batched_obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), raw_obs)

            action, rnn_state = policy_step(params, rnn_state, batched_obs)
            action_np = np.array(action).flatten()

            left_leg_torque = action_np[0:5]
            right_leg_torque = action_np[5:10]
            symmetry_delta = np.mean(np.abs(left_leg_torque - right_leg_torque))
            energy_draw = np.sum(np.square(action_np))

            robot_x = data.xpos[robot_body_id][0]
            if robot_x > last_robot_x + 0.5:
                success_streak += 1
                last_robot_x = robot_x
                if success_streak % 10 == 0:
                    curriculum_level = min(1.0, curriculum_level + 0.1)
            
            sys.stdout.write(
                f"\r[Telemetry] Step: {total_ticks:04d} | Level: {curriculum_level:.2f} | "
                f"Streak: {success_streak:02d} | "
                f"Symmetry Index: {symmetry_delta:.4f} | "
                f"Energy Draw: {energy_draw:.2f} W  "
            )
            sys.stdout.flush()
            total_ticks += 1

        # Apply action with the muscle multiplier scale (matches env.step scaling!)
        data.ctrl[:model.nu] = np.array(action).flatten() * 100.0

        for idx, joint_idx in enumerate(platform_joint_ids):
            base_offset = 0.5 * idx
            data.qpos[model.jnt_qposadr[joint_idx]] = base_offset * (1.0 + curriculum_level)

        mujoco.mj_step(model, data)
        step_counter += 1

        # Use absolute global Z coordinate of the robot
        robot_z_height = data.xpos[robot_body_id][2]
        if robot_z_height < 0.40:
            sys.stdout.write(f"\n⚠️ Robot fell (Z={robot_z_height:.2f}m)! Resetting...\n")
            sys.stdout.flush()
            reset_simulation_state()

        time_elapsed = time.time() - step_start
        if time_elapsed < 0.001:
            time.sleep(0.001 - time_elapsed)

        viewer.sync()