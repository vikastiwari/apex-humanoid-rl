# JAX & Flax Training Guide (Adapted from ASTRO_GUIDE)

> *Note: This document replaces the standard enterprise ASTRO_GUIDE, as this is a headless hardware-accelerated Machine Learning project.*

## Understanding JAX Vectorization (`vmap`)
In standard Python RL, you loop over environments. In JAX, we use `vmap` to vectorize the environment step function across the batch dimension.
```python
# Traditional approach (Slow)
for env in envs:
    env.step(action)

# JAX approach (Instantaneous on GPU)
vmap_step = jax.vmap(env.step, in_axes=(0, 0))
states = vmap_step(batched_states, batched_actions)
```
Ensure all tensors maintain consistent batch dimensions. If shapes mismatch, XLA compilation will fail instantly.

## Flax Neural Networks
We use Flax for defining the Actor-Critic networks.
- State is immutable. You must explicitly pass `params` and `rnn_states` into the `apply` function.
- Initialization requires a dummy observation tensor broadcasted to the correct batch size to trace the neural graph.

## Optax Optimizers
Gradients are calculated using `jax.value_and_grad`. Optax applies these gradients. Always remember to extract and pass the `opt_state` back through the training loop, as optimizers in JAX maintain pure functional state.
