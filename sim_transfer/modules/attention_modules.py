import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Callable

from haiku import MultiHeadAttention

def hk_layer_norm(*, axis, name=None):
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)

class ScoreNetworkAttentionModel(hk.Module):
    """ Attention-based architecture which uses a combined embedding on the input-output concatenation. """
    def __init__(self,
                 x_dim: int,
                 hidden_dim: int = 128,
                 layers: int = 8,
                 layer_norm: bool = True,
                 dropout_rate: float = 0.0,
                 layer_norm_axis: int = -1,
                 widening_factor: int = 2,
                 num_heads: int = 8,
                 key_size: int = 32,
                 logit_bias_init: float = -3.0,
                 fc_layer_activation_fn: Callable = jax.nn.gelu,
                 name: str = "ScoreNetworkAttentionModel",
                 ):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.layer_norm = layer_norm
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.layer_norm_axis = layer_norm_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
        self.fc_layer_activation_fn = fc_layer_activation_fn

    def __call__(self, x, is_training: bool = False):
        dropout_rate = self.dropout_rate if is_training else 0.0
        z = hk.Linear(self.hidden_dim)(x)
        for _ in range(self.layers):
            # Multi-head attention
            q_in = hk_layer_norm(axis=self.layer_norm_axis)(z) if self.layer_norm else z
            k_in = hk_layer_norm(axis=self.layer_norm_axis)(z) if self.layer_norm else z
            v_in = hk_layer_norm(axis=self.layer_norm_axis)(z) if self.layer_norm else z
            z_attn = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.hidden_dim,
            )(q_in, k_in, v_in)
            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)

            # fully connected layer
            z_in = hk_layer_norm(axis=self.layer_norm_axis)(z) if self.layer_norm else z

            z_ffn = hk.Sequential([
                hk.Linear(self.widening_factor * self.hidden_dim, w_init=self.w_init),
                self.fc_layer_activation_fn,
                hk.Linear(self.hidden_dim, w_init=self.w_init),
            ])(z_in)

            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)

        z = hk_layer_norm(axis=self.layer_norm_axis)(z) if self.layer_norm else z

        # estimated score
        if self.layer_norm:
            score = jnp.squeeze(hk.Sequential([
                hk_layer_norm(axis=self.layer_norm_axis),
                hk.Linear(1, w_init=self.w_init),
            ])(z))
        else:
            score = jnp.squeeze(hk.Sequential([
                hk.Linear(1, w_init=self.w_init),
            ])(z))
        return score

