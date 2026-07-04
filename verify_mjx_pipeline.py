import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from parkour_env import inject_parkour_obstacles, ParkourEnv

def setup_scene():
    """
    1 & 2: Asset Loading Verification & Procedural Terrains Integration.
    Loads Unitree H1 model and dynamically builds a parkour scene.
    """
    print("[*] Generating Procedural XML with H1 and Parkour Obstacles...")
    base_xml = "./assets/h1.xml"
    
    # Generate XML with 3 platforms (which implies 2 gaps between them)
    modified_xml = inject_parkour_obstacles(base_xml, num_obstacles=3)
    
    print("[*] Compiling MuJoCo Physics Model...")
    try:
        mj_model = mujoco.MjModel.from_xml_string(modified_xml)
        mjx_model = mjx.put_model(mj_model)
        print(" [✓] Compilation Successful!")
    except Exception as e:
        print(f" [!] Compilation Failed! Error: {e}")
        raise e
        
    return mjx_model

@jax.jit
def compute_slip(fx: jax.Array, fy: jax.Array, fz: jax.Array, mu: float = 1.0) -> tuple[jax.Array, jax.Array]:
    """
    3. Friction Cone & Slip Mathematics Verification.
    Computes slip coefficient S = sqrt(Fx^2 + Fy^2) / Fz.
    Flags slipping if S > mu.
    """
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    fz_safe = jnp.maximum(fz, epsilon)
    
    S = jnp.sqrt(fx**2 + fy**2) / fz_safe
    is_slipping = S > mu
    
    return S, is_slipping

@jax.jit
def dummy_step(state: mjx.Data, model: mjx.Model) -> mjx.Data:
    """
    Dummy step to advance simulation with zero controls.
    """
    return mjx.step(model, state)

def benchmark_pipeline(mjx_model: mjx.Model, num_envs: int = 1024, num_steps: int = 500):
    """
    4. MJX Parallel Compilation Benchmark.
    """
    # Detect platform to scale down on CPU and avoid freezing the run
    platform = jax.devices()[0].platform.lower()
    if platform == 'cpu':
        print("\n[!] WARNING: JAX is running on CPU. Scaling down benchmark parameters to prevent long wait times.")
        print("    (Running 1,024 complex environments on a laptop CPU can take minutes. Scaling to 64 envs, 100 steps.)")
        num_envs = 64
        num_steps = 100

    print(f"\n[*] Preparing Benchmark for {num_envs} Parallel Environments...")
    
    # Initialize batch of states
    rng = jax.random.PRNGKey(42)
    
    # Vmap make_data
    vmap_make_data = jax.vmap(mjx.make_data)
    batched_states = vmap_make_data(jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (num_envs, *x.shape)), mjx_model))
    
    # Set some initial random joint positions
    qpos = batched_states.qpos + jax.random.uniform(rng, (num_envs, mjx_model.nq), minval=-0.01, maxval=0.01)
    batched_states = batched_states.replace(qpos=qpos)
    
    # Vmap the step function
    vmap_step = jax.vmap(dummy_step, in_axes=(0, None))
    
    # Define rollout with jax.lax.scan to optimize execution and avoid Python loop dispatch overhead.
    # We specify static_argnums=(1,) because the loop length in jax.lax.scan must be a concrete Python integer at compile time.
    @jax.jit(static_argnums=(1,))
    def run_rollout(states, steps_count: int):
        def scan_fn(carry_state, _):
            next_state = vmap_step(carry_state, mjx_model)
            return next_state, None
        final_state, _ = jax.lax.scan(scan_fn, states, None, length=steps_count)
        return final_state
    
    print("[🚀] Starting JIT Compilation & Benchmark...")
    
    # 1. Compilation Time (Compiles the entire vmap + scan rollout)
    start_compile = time.time()
    # Trigger JIT compilation (we run for just 1 step to compile)
    compiled_state = run_rollout(batched_states, 1)
    jax.block_until_ready(compiled_state)
    compile_time = time.time() - start_compile
    print(f" [✓] JIT Compilation Time: {compile_time:.4f} seconds")
    
    # 2. Execution Throughput (SPS) using compiled lax.scan
    start_run = time.time()
    final_states = run_rollout(batched_states, num_steps)
    jax.block_until_ready(final_states)
    run_time = time.time() - start_run
    
    total_steps = num_envs * num_steps
    sps = total_steps / run_time
    
    print("\n================ BENCHMARK RESULTS ================")
    print(f"Platform:            {platform.upper()}")
    print(f"Environments:        {num_envs}")
    print(f"Steps per Env:       {num_steps}")
    print(f"Total Steps Executed:{total_steps}")
    print(f"Execution Time:      {run_time:.4f} seconds")
    print(f"Steps Per Second:    {sps:,.2f} SPS")
    print("===================================================\n")
    
    # Verify slip logic with dummy data
    print("[*] Verifying Slip Mathematics:")
    fx = jnp.array([0.1, 1.5, 0.5])
    fy = jnp.array([0.1, 1.0, 0.0])
    fz = jnp.array([10.0, 2.0, 0.0])
    mu = 0.8
    S, slip_flag = compute_slip(fx, fy, fz, mu)
    print(f" -> Forces Fz: {fz}")
    print(f" -> Slip Coefficient S: {S}")
    print(f" -> Slipping State (S > {mu}): {slip_flag}")

if __name__ == "__main__":
    mjx_model = setup_scene()
    benchmark_pipeline(mjx_model)
