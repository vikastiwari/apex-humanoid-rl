# Detailed Roadmap

The evolution of the Apex Humanoid project is split into distinct phases targeting robust sim-to-real transfer.

## Phase 1: Micro-locomotion (Complete)
- [x] Basic standing balance (see `train_standing_balance.py`)
- [x] Simple forward walking locomotion without falling
- [x] Basic MJX pipeline verification

## Phase 2: Dynamic Parkour (Current)
- [x] Vectorized PPO training loop
- [x] Random obstacle generation (hurdles, gaps)
- [ ] Achieve >90% success rate on 1-meter gaps
- [ ] Integrate advanced MCTS for multi-step jump planning

## Phase 3: Sim-to-Real Transfer (Upcoming)
- [ ] **Domain Randomization:** Randomize friction, mass, and motor torque limits during training to ensure the policy can handle physical world imperfections.
- [ ] **Actuator Modeling:** Replace ideal torque constraints with realistic motor models (e.g., modeling back-EMF and gear backlash).
- [ ] **Hardware Deployment:** Deploy the optimized XLA inference graph onto the physical H1 hardware via TensorRT.
