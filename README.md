# Apex Humanoid Parkour (apex-humanoid-rl)

An advanced Deep Reinforcement Learning pipeline utilizing JAX and MuJoCo MJX to train the H1 humanoid robot in complex parkour and locomotion tasks via massively parallel PPO.

## 🚀 Overview

The `apex-humanoid-rl` project is a cutting-edge robotics training framework. It leverages hardware-accelerated physics simulation (MuJoCo MJX) and fully vectorized JAX/Flax neural networks to train high-dimensional policies in absolute deterministic, microsecond-scale latency. 

By running thousands of environments natively on the GPU, we bypass the traditional CPU-GPU data transfer bottlenecks, allowing the H1 humanoid to learn complex behaviors—like balancing, running, and navigating parkour obstacles—in a fraction of the time required by standard CPU-based physics engines.

## 🧠 Core Architecture

- **Physics Engine:** MuJoCo MJX (JAX-native rigid body dynamics)
- **Neural Framework:** JAX, Flax, Optax
- **Algorithm:** Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
- **Environment:** Custom `ParkourEnv` mapping H1 joint positions and sensor fusions to complex reward landscapes.

## 📁 Repository Structure

- `train_parkour.py`: The massively parallel JAX-JIT compiled PPO training loop.
- `parkour_env.py`: The core MuJoCo MJX environment logic, reward functions, and obstacle generation.
- `sensor_fusion.py`: The Actor-Critic neural network architecture processing proprioceptive and exteroceptive data.
- `mcts_planner.py`: Monte Carlo Tree Search trajectory planner for advanced heuristic lookahead.
- `render_parkour.py`: Visual debugger to render the trained policies and observe physical interactions.

## 📖 Documentation

For a deep dive into the engineering paradigms, consult the `/docs` directory:
- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [System Components](docs/COMPONENTS.md)
- [JAX & Flax Training Guide (Astro Guide)](docs/ASTRO_GUIDE.md)
- [Simulation Rendering Design (UI/UX)](docs/UI_UX_DESIGN.md)
- [Detailed Roadmap](docs/DETAILED_ROADMAP.md)

## 🛠️ Quick Start

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
*(Ensure you have a CUDA-compatible GPU setup for JAX hardware acceleration)*

### 2. Verify MJX Pipeline
```bash
python3 verify_mjx_pipeline.py
```

### 3. Launch Training
```bash
python3 train_parkour.py
```

### 4. Render Policy
```bash
python3 render_parkour.py
```

## ⚖️ License
Enterprise Proprietary. All rights reserved.
