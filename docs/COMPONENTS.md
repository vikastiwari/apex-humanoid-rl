# System Components

This document outlines the core files and classes that construct the Apex humanoid pipeline.

## 1. `parkour_env.py`
**Responsibility:** Simulates the physical world.
- Uses `mujoco.mjx` for XLA-compiled physics computations.
- Contains the `ParkourEnv` class, responsible for parsing the `h1.xml` model, calculating dense rewards (forward velocity, upright posture, energy penalty), and determining termination states (e.g., falling over).

## 2. `sensor_fusion.py`
**Responsibility:** Artificial Brain (Neural Network).
- Uses Flax `nn.Module` to define the architecture.
- Maps continuous observations (joint angles, velocities) into continuous actions (motor torques) using a multi-layer perceptron (MLP).
- Contains separate heads for the Actor (policy variance/mean) and the Critic (value estimation).

## 3. `train_parkour.py`
**Responsibility:** Orchestration & Optimization.
- Sets up XLA vectorized environments (`jax.vmap`).
- Performs Generalized Advantage Estimation (GAE) to evaluate action quality.
- Uses Optax to backpropagate gradients and update network parameters using the clipped PPO surrogate objective.

## 4. `mcts_planner.py`
**Responsibility:** Advanced Heuristic Lookahead.
- A discrete Monte Carlo Tree Search module designed to inject high-level strategic planning into the continuous control problem.

## 5. `render_parkour.py`
**Responsibility:** Visual Debugging.
- Reconstructs the trained parameters inside the standard CPU `mujoco.viewer`.
- Allows human engineers to observe the physical accuracy of the policies in real-time.
