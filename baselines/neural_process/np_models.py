from flax import linen as nn
from typing import List

import jax.numpy as jnp
import jax

class MLP(nn.Module):
    hidden_layer_sizes: List[int]
    output_size: int

    def setup(self):
        self.hidden_layers = [nn.Dense(layer) for layer in self.hidden_layer_sizes]
        self.out_layer = nn.Dense(self.output_size)

    def __call__(self, x):
        for layer in self.hidden_layers[:-1]:
            x = nn.leaky_relu(layer(x))
        return self.out_layer(x)


class TransformerModel(nn.Module):
    output_dim: int
    num_heads: int
    hidden_dim: int = 32
    num_blocks: int = 2

    def setup(self):
        self.dense_in = nn.Dense(self.hidden_dim)

        self.transformer_blocks = [{
            'attn': nn.attention.SelfAttention(num_heads=self.num_heads),
            'linear': [
                nn.Dense(self.hidden_dim),
                nn.leaky_relu,
                nn.Dense(self.hidden_dim)
            ],
            'norm1': nn.LayerNorm(),
            'norm2': nn.LayerNorm()
        } for _ in range(self.num_blocks)]

        self.dense_out = nn.Dense(self.output_dim)

    def __call__(self, x, train=True):
        x = self.dense_in(x)

        for block_modules in self.transformer_blocks:
            # Attention part
            attn_out = block_modules['attn'](x)
            # x = x + self.dropout(attn_out, deterministic=not train)
            x = x + attn_out
            x = block_modules['norm1'](x)

            # MLP part
            linear_out = x
            for l in block_modules['linear']:
                linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
            # x = x + self.dropout(linear_out, deterministic=not train)
            x = x + linear_out
            x = block_modules['norm2'](x)

        x = self.dense_out(x)
        return x


class NPDecoder(nn.Module):
    hidden_layer_sizes: List[int]
    output_size: int

    def setup(self):
        self.mlp = MLP(self.hidden_layer_sizes, output_size=2 * self.output_size)

    def __call__(self, x):
        mlp_out = self.mlp(x)
        mu, log_sigma = jnp.split(mlp_out, 2, axis=-1)
        # Lower bound the variance
        sigma = 0.1 + 0.9 * nn.softplus(log_sigma)
        return mu, sigma

def dot_product_attn(query, keys, values):
    return jnp.sum(nn.softmax(jnp.sum(query[..., None, :] * keys, axis=-1) /
                       jnp.sqrt(keys.shape[-1]), axis=-1)[..., None] * values, axis=-2)


class NPEncoderDet(nn.Module):
    latent_dim: int
    hidden_dim: list = 32
    num_layers: int = 3
    use_cross_attention: bool = False
    use_self_attention: bool = False
    num_attn_heads: int = 4
    num_transf_blocks: int = 2

    def setup(self):
        if self.use_self_attention:
            self.enc_det = TransformerModel(output_dim=self.latent_dim, num_heads=self.num_attn_heads,
                                            hidden_dim=self.hidden_dim, num_blocks=self.num_transf_blocks)
        else:
            self.enc_det = MLP([self.hidden_dim] * self.num_layers, output_size=self.latent_dim)

        if self.use_cross_attention:
            self.key_mlp = MLP(hidden_layer_sizes=[self.hidden_dim], output_size=self.latent_dim)
            self.query_mlp = MLP(hidden_layer_sizes=[self.hidden_dim], output_size=self.latent_dim)

    def __call__(self, x, y, x_target):
        batch_shape = x.shape[:-2]
        num_target_points = x_target.shape[-2]
        xy = jnp.concatenate([x, y], -1)

        # deterministic encoder
        r_i = self.enc_det(xy)
        if self.use_cross_attention:
            keys = self.key_mlp(x)
            query = self.query_mlp(x_target)
            r_agg = jax.vmap(dot_product_attn, in_axes=(-2, None, None), out_axes=-2)(query, keys, r_i)
        else:
            r_agg = jnp.repeat(jnp.mean(r_i, axis=-2)[..., None, :], num_target_points, axis=-2)
        return r_agg


class NPEncoderStoch(nn.Module):
    latent_dim: int
    hidden_dim: list = 32
    num_layers: int = 3
    use_self_attention: bool = False
    num_attn_heads: int = 4
    num_transf_blocks: int = 2

    def setup(self):
        if self.use_self_attention:
            self.enc_stoch = TransformerModel(output_dim=2*self.latent_dim, num_heads=self.num_attn_heads,
                                            hidden_dim=self.hidden_dim, num_blocks=self.num_transf_blocks)
        else:
            self.enc_stoch = MLP([self.hidden_dim] * self.num_layers, output_size=2*self.latent_dim)

    def __call__(self, x, y):
        xy = jnp.concatenate([x, y], -1)

        # stochastic encoder
        s_i = self.enc_stoch(xy)
        sc = jnp.mean(s_i, axis=-2)
        mu, log_sigma = jnp.split(sc, 2, axis=-1)
        sigma = 0.1 + 0.9 * nn.softplus(log_sigma)
        return mu, sigma

