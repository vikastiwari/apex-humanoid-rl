# Known Bugs & Constraints

## 1. XLA Compilation Timeouts
When increasing `num_envs` beyond 4096 in `train_parkour.py`, the initial JAX JIT compilation step can take several minutes or time out entirely.
**Workaround:** Ensure you have enough system RAM before launching large batches, as XLA requires significant host memory during graph construction.

## 2. Policy Collapse (Reward Hacking)
In the current iteration, the policy occasionally converges into a local minimum where the H1 robot stands still to avoid energy penalties instead of jumping over obstacles.
**Workaround:** We are continuously tuning the `c_ent` (entropy coefficient) in `ppo_loss` to force exploration, and adjusting the forward velocity reward scaling in `ParkourEnv`.

## 3. MJX Collision Penetrations
Due to soft-contact solver parameters in MuJoCo MJX, high-velocity impacts can cause the robot's feet to penetrate the ground plane.
**Workaround:** Tighten the solver iterations or increase the stiffness parameters in `assets/h1.xml`.
