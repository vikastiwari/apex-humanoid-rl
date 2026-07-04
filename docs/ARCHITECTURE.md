# System Architecture

The `apex-humanoid-rl` system is designed around extreme parallelism, utilizing hardware acceleration (GPUs) to compute thousands of physics simulations and neural network passes simultaneously.

## Core Architectural Pillars

### 1. Vectorized Physics (MuJoCo MJX)
Traditional RL pipelines suffer from a massive bottleneck: the physics engine runs on the CPU, while the neural network runs on the GPU. This requires transferring state data and actions across the PCIe bus millions of times, slowing down training.
Our architecture leverages **MuJoCo MJX**, which compiles the rigid-body physics calculations directly into JAX. This allows the entire physics simulation to run natively on the GPU, completely eliminating the PCIe bottleneck.

### 2. JAX-JIT Compiled Training Loop
The entire Proximal Policy Optimization (PPO) step—from trajectory rollout to Generalized Advantage Estimation (GAE) and Optax gradient updates—is wrapped in `jax.jit`. 
This triggers XLA (Accelerated Linear Algebra) to compile the Python logic into highly optimized fused GPU kernels.

### 3. Sensor Fusion Actor-Critic
The intelligence of the H1 robot is governed by the `SensorFusionActorCritic` network (implemented via Flax). 
- **Proprioception:** Joint angles, angular velocities, and IMU data.
- **Exteroception:** Lidar/depth estimation to understand obstacle proximity.
The architecture uses continuous action spaces (diagonal Gaussian distributions) to output high-frequency torque commands to the robot's motors.

## Control Flow
1. **Reset Phase:** `jax.vmap(env.reset)` spawns 1,024 independent environments simultaneously.
2. **Rollout Phase:** The Actor-Critic model queries the environment state and selects actions. 
3. **Step Phase:** `jax.vmap(env.step)` executes physics simulation for all 1,024 environments in parallel.
4. **Update Phase:** GAE is calculated backwards in time; PPO clipped surrogate loss updates the neural weights.
5. **Curriculum Phase:** As the robot's success rate increases, the environment dynamically injects more difficult parkour obstacles.
