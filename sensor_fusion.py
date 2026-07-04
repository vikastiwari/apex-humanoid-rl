import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any

class TactileEncoder(nn.Module):
    """
    1D CNN to process high-frequency foot contact data.
    Input shape expects [batch_size, seq_len, features].
    """
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, tactile_data: jax.Array) -> jax.Array:
        # Example: tactile_data is [batch, seq_len, 6] (3 features per foot)
        # Apply 1D Convolutions across the sequence (temporal or spatial window)
        # Assuming input here for a single timestep is reshaped to (batch, 1, 6) or processed as dense.
        # For standard RL steps without explicit temporal windowing in the obs, we treat it as an MLP for the current step.
        # However, a 1D CNN would be used if passing a sliding window of tactile history.
        
        # Scaffold as an MLP for single-step feature extraction before the GRU
        x = nn.Dense(self.hidden_dim)(tactile_data)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return x


class ProprioceptiveEncoder(nn.Module):
    """
    MLP to process joint states and kinematic data.
    """
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, proprio_data: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim)(proprio_data)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return x


class ExteroceptiveEncoder(nn.Module):
    """
    MLP to process Raycast/Lidar data.
    """
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, extero_data: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim)(extero_data)
        x = nn.relu(x)
        return x


class WaypointEncoder(nn.Module):
    """
    MLP to process high-level 3D target waypoint vectors.
    """
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, waypoint_data: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim)(waypoint_data)
        x = nn.relu(x)
        return x


class SensorFusionActorCritic(nn.Module):
    """
    Tactile-Proprioceptive Sensor Fusion Network.
    Encodes multi-modal observations, fuses them, applies GRU for memory, 
    and outputs Actor (action distribution) and Critic (value) components.
    """
    action_dim: int
    rnn_hidden_dim: int = 256

    @nn.compact
    def __call__(self, rnn_state: jax.Array, obs: Dict[str, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Forward pass for the Actor-Critic.
        
        Args:
            rnn_state: Hidden state for the GRU cell [batch_size, rnn_hidden_dim].
            obs: Dictionary of observation arrays.
            
        Returns:
            new_rnn_state: Updated hidden state for the GRU cell.
            action_mean: Mean of the action distribution.
            action_log_std: Trainable log standard deviation.
            value: Estimated state value.
        """
        # 1. Independent Encoders
        tactile_feat = TactileEncoder()(obs["tactile"])
        proprio_feat = ProprioceptiveEncoder()(obs["proprioception"])
        extero_feat = ExteroceptiveEncoder()(obs["exteroception"])
        waypoint_feat = WaypointEncoder()(obs["waypoint_vector"])
        
        # 2. Fusion Layer
        # Concatenate all encoded streams including the high-level MCTS targets
        fused_features = jnp.concatenate([tactile_feat, proprio_feat, extero_feat, waypoint_feat], axis=-1)
        
        # 3. GRU for Temporal Memory (Crucial for slip recovery in < 5ms)
        # We use flax.linen.GRUCell.
        gru_cell = nn.GRUCell(features=self.rnn_hidden_dim)
        
        # GRUCell expects inputs of shape (batch, features).
        # It returns (new_carry, output), where output is same as new_carry.
        new_rnn_state, gru_out = gru_cell(rnn_state, fused_features)
        
        # 4. Actor Head (Policy)
        # Outputs the mean of a continuous Gaussian policy
        action_mean = nn.Dense(128)(gru_out)
        action_mean = nn.relu(action_mean)
        action_mean = nn.Dense(self.action_dim)(action_mean)
        
        # Trainable log standard deviation parameter for continuous policy
        action_log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        
        # 5. Critic Head (Value Function)
        value = nn.Dense(128)(gru_out)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        
        # Squeeze value to shape (batch,)
        value = jnp.squeeze(value, axis=-1)
        
        return new_rnn_state, action_mean, action_log_std, value
