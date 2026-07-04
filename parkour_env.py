import os
import xml.etree.ElementTree as ET
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import flax.struct
from typing import Dict, Any, Tuple

def inject_parkour_obstacles(base_xml_path: str, num_obstacles: int = 5) -> str:
    """
    Dynamically modifies the base H1 XML to include parkour obstacles (gaps, varying height platforms).
    """
    import numpy as np
    try:
        tree = ET.parse(base_xml_path)
        root = tree.getroot()
    except FileNotFoundError:
        # Fallback to a basic template if h1.xml is not found locally for scaffolding purposes
        root = ET.fromstring('''<mujoco model="h1_parkour">
            <worldbody>
                <light name="top" pos="0 0 1.5"/>
                <geom name="floor" type="plane" size="10 10 0.1" rgba=".8 .9 .8 1"/>
                <body name="torso" pos="0 0 1">
                    <joint name="root" type="free"/>
                    <geom name="torso_geom" type="box" size="0.1 0.1 0.2" mass="10"/>
                    <body name="left_leg" pos="0.1 0 -0.2">
                        <joint name="l_hip" type="hinge" axis="0 1 0"/>
                        <geom name="left_foot" type="box" size="0.05 0.05 0.05" pos="0 0 -0.4"/>
                    </body>
                    <body name="right_leg" pos="-0.1 0 -0.2">
                        <joint name="r_hip" type="hinge" axis="0 1 0"/>
                        <geom name="right_foot" type="box" size="0.05 0.05 0.05" pos="0 0 -0.4"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>''')

    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')
        
    # Resolve meshdir relative to the XML file location to prevent path errors
    import os
    compiler = root.find('compiler')
    if compiler is not None and 'meshdir' in compiler.attrib:
        original_meshdir = compiler.attrib['meshdir']
        xml_dir = os.path.dirname(os.path.abspath(base_xml_path))
        abs_meshdir = os.path.join(xml_dir, original_meshdir)
        compiler.attrib['meshdir'] = abs_meshdir

    # Inject global directional light for proper 3D rendering illumination
    ET.SubElement(worldbody, 'light', {
        'directional': 'true',
        'diffuse': '0.8 0.8 0.8',
        'specular': '0.2 0.2 0.2',
        'pos': '0 0 10',
        'dir': '0 -1 -1'
    })

    # Improve rendering visibility: Make robot lighter
    asset_elem = root.find('asset')
    if asset_elem is None:
        asset_elem = ET.SubElement(root, 'asset')
    
    ET.SubElement(asset_elem, 'texture', {
        'name': 'grid',
        'type': '2d',
        'builtin': 'checker',
        'rgb1': '.1 .2 .3',
        'rgb2': '.2 .3 .4',
        'width': '512',
        'height': '512',
        'mark': 'cross',
        'markrgb': '.8 .8 .8'
    })
    ET.SubElement(asset_elem, 'material', {
        'name': 'grid_mat',
        'texture': 'grid',
        'texrepeat': '40 40',
        'texuniform': 'true'
    })

    # Add basic floor with grid texture
    floor_elem = worldbody.find("geom[@name='floor']")
    if floor_elem is not None:
        worldbody.remove(floor_elem)

    ET.SubElement(worldbody, 'geom', {
        'name': 'floor',
        'type': 'plane',
        'size': '40 40 0.1',
        'material': 'grid_mat'
    })

    ET.SubElement(asset_elem, 'texture', {
        'type': 'skybox',
        'builtin': 'gradient',
        'rgb1': '0.4 0.6 0.8',
        'rgb2': '0.1 0.1 0.15',
        'width': '512',
        'height': '512'
    })

    # Make the default 'black' material lighter so the robot isn't too dark
    for mat in root.iter('material'):
        if mat.get('name') == 'black':
            mat.set('rgba', '0.8 0.8 0.8 1')

    # Convert all cylinder geometries to capsules because MJX does not support cylinder-box collisions
    for geom in root.iter('geom'):
        if geom.get('type') == 'cylinder':
            geom.set('type', 'capsule')

    # Dynamically name the foot collision geometries so we can extract their contact forces in MJX
    for body in root.iter('body'):
        if body.get('name') == 'left_ankle_link':
            for i, geom in enumerate(body.findall('geom')):
                if geom.get('class') != 'visual':
                    geom.set('name', f'left_foot_{i}')
        elif body.get('name') == 'right_ankle_link':
            for i, geom in enumerate(body.findall('geom')):
                if geom.get('class') != 'visual':
                    geom.set('name', f'right_foot_{i}')

    # Procedurally add platforms
    np.random.seed(42)
    current_x = 1.0
    for i in range(num_obstacles):
        # Randomize gap (distance to next platform) and platform height
        gap = np.random.uniform(0.5, 1.5)
        height = np.random.uniform(-0.2, 0.2)
        platform_length = np.random.uniform(1.0, 2.0)
        
        current_x += gap + (platform_length / 2)
        
        body = ET.SubElement(worldbody, 'body', {
            'name': f'platform_{i}_body',
            'pos': f'{current_x} 0 {height - 0.5}'
        })
        ET.SubElement(body, 'joint', {
            'name': f'platform_{i}_x',
            'type': 'slide',
            'axis': '1 0 0',
            'damping': '10000',
            'frictionloss': '10000'
        })
        ET.SubElement(body, 'joint', {
            'name': f'platform_{i}_z',
            'type': 'slide',
            'axis': '0 0 1',
            'damping': '10000',
            'frictionloss': '10000'
        })
        ET.SubElement(body, 'geom', {
            'name': f'platform_{i}',
            'type': 'box',
            'size': f'{platform_length/2} 1.0 0.5',
            'rgba': '0.5 0.5 0.8 1',
            'condim': '3',
            'friction': '1 0.005 0.0001',
            'mass': '1000'
        })
        current_x += (platform_length / 2)

    # Inject framepos sensor for absolute robust tracking of the pelvis
    sensor_elem = root.find('sensor')
    if sensor_elem is None:
        sensor_elem = ET.SubElement(root, 'sensor')
    ET.SubElement(sensor_elem, 'framepos', {
        'name': 'pelvis_position',
        'objtype': 'body',
        'objname': 'pelvis'
    })

    return ET.tostring(root, encoding='unicode')

def compute_slip(fx: jax.Array, fy: jax.Array, fz: jax.Array, mu: float = 1.0) -> tuple[jax.Array, jax.Array]:
    """
    Computes slip coefficient S = sqrt(Fx^2 + Fy^2) / Fz.
    Flags slipping if S > mu.
    """
    epsilon = 1e-6
    fz_safe = jnp.maximum(fz, epsilon)
    S = jnp.sqrt(fx**2 + fy**2) / fz_safe
    is_slipping = S > mu
    return S, is_slipping

@flax.struct.dataclass
class EnvState:
    physics: mjx.Data
    mjx_model: mjx.Model
    curriculum_level: jax.Array
    success_streak: jax.Array
    last_action: jax.Array
    step_counter: jax.Array
    last_progress_x: jax.Array
    rng: jax.Array

class ParkourEnv:
    """
    Custom MuJoCo MJX Environment for Project APEX.
    Simulates a humanoid (Unitree H1) traversing a procedurally generated parkour course.
    """

    def __init__(self, base_xml_path: str = "h1.xml", num_obstacles: int = 5):
        """
        Initializes the environment and procedurally generates the XML.
        """
        self.num_obstacles = num_obstacles
        self.base_xml_path = base_xml_path
        
        # 1. Generate the modified XML with parkour obstacles
        self.xml_string = inject_parkour_obstacles(self.base_xml_path, self.num_obstacles)
        
        # 2. Compile the MuJoCo model and create MJX model
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.mjx_model = mjx.put_model(self.mj_model)

        # Resolve body ID for pelvis (for posture calculation)
        try:
            self._pelvis_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        except Exception:
            self._pelvis_body_id = 1 # Fallback to body 1 (root pelvis)

        # Dynamically look up the geom IDs of the renamed feet collision shapes
        self._left_foot_geom_ids = []
        self._right_foot_geom_ids = []
        for i in range(10):
            lf_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_foot_{i}")
            if lf_id != -1:
                self._left_foot_geom_ids.append(lf_id)
            rf_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"right_foot_{i}")
            if rf_id != -1:
                self._right_foot_geom_ids.append(rf_id)
                
        # Extract platform joint indices for dynamic curriculum resets
        self._platform_x_qpos_idx = []
        self._platform_z_qpos_idx = []
        for i in range(self.num_obstacles):
            x_jnt = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f'platform_{i}_x')
            z_jnt = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f'platform_{i}_z')
            if x_jnt != -1:
                self._platform_x_qpos_idx.append(self.mj_model.jnt_qposadr[x_jnt])
            if z_jnt != -1:
                self._platform_z_qpos_idx.append(self.mj_model.jnt_qposadr[z_jnt])
        self._platform_x_qpos_idx = jnp.array(self._platform_x_qpos_idx, dtype=jnp.int32)
        self._platform_z_qpos_idx = jnp.array(self._platform_z_qpos_idx, dtype=jnp.int32)
        
        # Configuration parameters for dense multi-objective rewards (easily tunable)
        self.config = {
            "w_forward": 10.0,      # Reward for forward velocity (vx)
            "w_survival": 2.0,     # Pos reward for keeping torso above min height
            "w_energy": -0.005,    # Penalty for large action outputs (torques)
            "w_posture": -1.5,     # Penalty for torso tilting (keeping robot upright)
            "w_slip": -5.0,        # Heavy penalty for slip events (force weight distribution)
            "w_waypoint_tracking": -3.0, # Heavy penalty for distance to MCTS waypoint
            "survival_height": 0.5, # Z-height threshold of pelvis to survive
            "friction_mu": 0.8,     # Simulated static friction limit
        }

    def get_foot_forces(self, state: mjx.Data, foot_geom_ids: list) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        JIT-compatible extraction of contact forces on the feet geometries from MJX.
        """
        geom1 = state.contact.geom1
        geom2 = state.contact.geom2
        efc_address = state.contact.efc_address
        efc_force = state.efc_force
        
        # Verify active contacts using JAX-compatible index bounds
        idx = jnp.arange(geom1.shape[0])
        is_active = idx < state.ncon
        
        # Build mask for contacts matching the foot geoms
        in_foot = jnp.zeros_like(geom1, dtype=jnp.bool_)
        for gid in foot_geom_ids:
            in_foot = in_foot | (geom1 == gid) | (geom2 == gid)
            
        mask = is_active & in_foot
        
        # Clamp index references to prevent out-of-bounds compiles in JAX
        safe_address = jnp.where(mask & (efc_address >= 0), efc_address, 0)
        
        # Sum forces (normal is at efc_address, shear is at +1 and +2)
        fz = jnp.sum(jnp.where(mask & (efc_address >= 0), efc_force[safe_address], 0.0))
        fx = jnp.sum(jnp.where(mask & (efc_address >= 0), efc_force[safe_address + 1], 0.0))
        fy = jnp.sum(jnp.where(mask & (efc_address >= 0), efc_force[safe_address + 2], 0.0))
        
        return fx, fy, fz

    def reset_physics(self, rng: jax.Array, curriculum_level: jax.Array) -> Tuple[mjx.Model, mjx.Data, Dict[str, jax.Array]]:
        rng, rng_mass, rng_fric = jax.random.split(rng, 3)
        
        # Layer 2: Extreme Domain Randomization
        mass_scale = jax.random.uniform(rng_mass, (self.mjx_model.body_mass.shape[0],), minval=0.9, maxval=1.1)
        new_body_mass = self.mjx_model.body_mass * mass_scale
        
        fric_scale = jax.random.uniform(rng_fric, (self.mjx_model.geom_friction.shape[0],), minval=0.7, maxval=1.3)
        new_geom_friction = self.mjx_model.geom_friction * jnp.expand_dims(fric_scale, axis=-1)
        
        randomized_model = self.mjx_model.replace(body_mass=new_body_mass, geom_friction=new_geom_friction)
        
        state = mjx.make_data(randomized_model)
        
        # Add random noise to initial position/velocity for robustness
        qpos = randomized_model.qpos0 + jax.random.uniform(rng, (randomized_model.nq,), minval=-0.01, maxval=0.01)
        qvel = jax.random.uniform(rng, (randomized_model.nv,), minval=-0.01, maxval=0.01)
        
        # Slider Joint Strategy: Manipulate platform joints directly in qpos based on curriculum_level
        rng_z, rng_x = jax.random.split(rng)
        
        # X gap: up to 1.5m at level 1.0 (approximated with cumsum offset)
        x_offsets = jax.random.uniform(rng_x, (self.num_obstacles,), minval=0.0, maxval=1.5 * curriculum_level)
        x_shift = jnp.cumsum(x_offsets)
        
        # Z height: +/- 0.2m scaled by curriculum_level
        z_offsets = jax.random.uniform(rng_z, (self.num_obstacles,), minval=-0.2 * curriculum_level, maxval=0.2 * curriculum_level)
        
        if len(self._platform_x_qpos_idx) > 0:
            qpos = qpos.at[self._platform_x_qpos_idx].set(x_shift)
            qpos = qpos.at[self._platform_z_qpos_idx].set(z_offsets)
            
        state = state.replace(qpos=qpos, qvel=qvel)
        state = mjx.step(randomized_model, state)
        obs = self.get_observation(state, jnp.zeros(3))
        return randomized_model, state, obs

    def reset(self, rng: jax.Array) -> Tuple[EnvState, Dict[str, jax.Array]]:
        randomized_model, physics, obs = self.reset_physics(rng, jnp.array(0.0))
        env_state = EnvState(
            physics=physics,
            mjx_model=randomized_model,
            curriculum_level=jnp.array(0.0),
            success_streak=jnp.array(0),
            last_action=jnp.zeros(self.mjx_model.nu),
            step_counter=jnp.array(0),
            last_progress_x=physics.qpos[0],
            rng=rng
        )
        return env_state, obs

    def get_observation(self, state: mjx.Data, target_waypoint: jax.Array = jnp.zeros(3)) -> Dict[str, jax.Array]:
        qpos = state.qpos
        qvel = state.qvel
        target_velocity_x = jnp.array([1.5])
        proprioception = jnp.concatenate([qpos, qvel, target_velocity_x])

        fx_l, fy_l, fz_l = self.get_foot_forces(state, self._left_foot_geom_ids)
        fx_r, fy_r, fz_r = self.get_foot_forces(state, self._right_foot_geom_ids)
        
        norm = 500.0
        tactile_left = jnp.array([fz_l / norm, fx_l / norm, fy_l / norm])  
        tactile_right = jnp.array([fz_r / norm, fx_r / norm, fy_r / norm])
        tactile = jnp.concatenate([tactile_left, tactile_right])
        
        exteroception = jnp.zeros((10,))
        pelvis_pos = state.qpos[0:3]
        waypoint_vector = target_waypoint - pelvis_pos

        return {
            "proprioception": proprioception,
            "tactile": tactile,
            "exteroception": exteroception,
            "waypoint_vector": waypoint_vector
        }

    @jax.jit
    def step(self, env_state: EnvState, action: jax.Array, target_waypoint: jax.Array = jnp.zeros(3)) -> Tuple[EnvState, Dict[str, jax.Array], jax.Array, jax.Array, Dict[str, jax.Array]]:
        # Residual Joint Position Control
        q_nominal = env_state.mjx_model.qpos0[7:7+env_state.mjx_model.nu]
        q_target = q_nominal + (action * 0.25)
        state = env_state.physics.replace(ctrl=q_target)
        new_physics = mjx.step(env_state.mjx_model, state)
        obs = self.get_observation(new_physics, target_waypoint)
        
        vx = new_physics.qvel[0]
        target_velocity_x = 1.5
        r_track = 5.0 * jnp.exp(-jnp.abs(vx - target_velocity_x))
        r_forward = self.config["w_forward"] * vx + r_track
        
        pelvis_height = new_physics.qpos[2]
        is_alive = pelvis_height > self.config["survival_height"]
        
        # Layer 6: Whole Body Control (WBC) Safety Constraints
        vel_penalty = jnp.sum(jnp.square(new_physics.qvel[6:]))
        r_vel = -0.001 * vel_penalty
        
        action_penalty = jnp.sum(jnp.square(action))
        r_energy = -0.005 * action_penalty
        
        leg_diff = action[0:5] - action[5:10]
        arm_diff = action[11:15] - action[15:19]
        r_symmetry = -0.05 * (jnp.sum(jnp.abs(leg_diff)) + jnp.sum(jnp.abs(arm_diff)))
        
        smoothness_penalty = jnp.sum(jnp.square(action - env_state.last_action))
        r_smooth = -0.01 * smoothness_penalty
        
        # Upright Posture Reward
        pelvis_pitch_approx = new_physics.xmat[self._pelvis_body_id, 0, 2]
        pelvis_roll_approx = new_physics.xmat[self._pelvis_body_id, 1, 2]
        r_posture = 5.0 * jnp.exp(-jnp.abs(pelvis_pitch_approx) - jnp.abs(pelvis_roll_approx))
        
        fx_l, fy_l, fz_l = self.get_foot_forces(new_physics, self._left_foot_geom_ids)
        fx_r, fy_r, fz_r = self.get_foot_forces(new_physics, self._right_foot_geom_ids)
        _, slip_l = compute_slip(fx_l, fy_l, fz_l, self.config["friction_mu"])
        _, slip_r = compute_slip(fx_r, fy_r, fz_r, self.config["friction_mu"])
        is_slipping = slip_l | slip_r
        r_slip = self.config["w_slip"] * jnp.where(is_slipping, 1.0, 0.0)
        
        pelvis_pos = new_physics.qpos[0:3]
        dist_to_waypoint = jnp.linalg.norm(pelvis_pos - target_waypoint)
        
        # Layer 1: Curriculum Annealing & Massive Survival Reward
        w_tracking = 1.0 + 14.0 * env_state.curriculum_level
        r_survival = jnp.where(is_alive, 10.0, 0.0)
        r_waypoint = -w_tracking * dist_to_waypoint
        
        total_reward = r_forward + r_survival + r_energy + r_symmetry + r_smooth + r_posture + r_slip + r_waypoint + r_vel
        
        # Locomotion progress "stick" penalty
        pelvis_pos = new_physics.qpos[0:3]
        new_step_counter = env_state.step_counter + 1
        check_progress = (new_step_counter % 100) == 0
        progress_made = (pelvis_pos[0] - env_state.last_progress_x) >= 0.05
        r_stuck = jnp.where(check_progress & ~progress_made, -50.0, 0.0)
        
        total_reward += r_stuck
        
        new_last_progress_x = jnp.where(check_progress, pelvis_pos[0], env_state.last_progress_x)
        
        done = jnp.where(is_alive, 0.0, 1.0)
        
        # Curriculum Advancement Logic
        reached_waypoint = dist_to_waypoint < 0.3
        new_streak = jnp.where(reached_waypoint, env_state.success_streak + 1, 
                       jnp.where(done, 0, env_state.success_streak))
        advance = new_streak >= 200
        new_level = jnp.minimum(1.0, env_state.curriculum_level + jnp.where(advance, 0.05, 0.0))
        new_streak = jnp.where(advance, 0, new_streak)
        
        rng, reset_rng = jax.random.split(env_state.rng)
        
        new_env_state = EnvState(
            physics=new_physics,
            mjx_model=env_state.mjx_model,
            curriculum_level=new_level,
            success_streak=new_streak,
            last_action=action,
            step_counter=new_step_counter,
            last_progress_x=new_last_progress_x,
            rng=rng
        )
        
        # Auto-reset
        def do_reset(_):
            reset_model, reset_physics, reset_obs = self.reset_physics(reset_rng, new_level)
            return new_env_state.replace(
                physics=reset_physics, 
                mjx_model=reset_model, 
                last_action=jnp.zeros_like(action),
                step_counter=jnp.array(0),
                last_progress_x=reset_physics.qpos[0]
            ), reset_obs
            
        def do_continue(_):
            return new_env_state, obs
            
        final_env_state, final_obs = jax.lax.cond(done, do_reset, do_continue, operand=None)
        
        info = {
            "reward_forward": r_forward,
            "reward_survival": r_survival,
            "reward_energy": r_energy,
            "reward_symmetry": r_symmetry,
            "reward_smooth": r_smooth,
            "reward_waypoint": r_waypoint,
            "curriculum_level": final_env_state.curriculum_level
        }
        
        return final_env_state, final_obs, total_reward, done, info
