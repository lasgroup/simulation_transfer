import math
from functools import partial
from typing import Optional, Sequence, Callable, Union, List, Dict

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.core import FrozenDict
from jax import random, vmap, jit
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_leaves

from sim_transfer.modules.util import RngKeyMixin


class BatchedModule(RngKeyMixin):
    def __init__(self, base_module: nn.Module, num_batched_modules: int, rng_key: random.PRNGKey):
        super().__init__(rng_key)
        self.base_module = base_module
        self.num_batched_modules = num_batched_modules

        self._param_vectors_stacked = self.get_init_param_vec_stacked()
        self.flatten_batch, self.unravel_batch = self.flatten_params()

    def flatten_params(self):
        one_batch = self._init_params_one_model(random.PRNGKey(0))
        flat_params, unravel_callable = ravel_pytree(one_batch)

        def only_flatten(params):
            return ravel_pytree(params)[0]

        flatten_batch = jit(vmap(only_flatten))
        unravel_batch = jit(vmap(unravel_callable))
        return flatten_batch, unravel_batch

    @property
    def param_vectors_stacked(self) -> jnp.ndarray:
        return self._param_vectors_stacked

    @param_vectors_stacked.setter
    def param_vectors_stacked(self, params_stacked: Union[List, Dict]):
        self._param_vectors_stacked = params_stacked

    def _init_params_one_model(self, key):
        variables = self.base_module.init(key, jnp.ones(shape=(self.base_module.input_size,)))
        # Split state and params (which are updated by optimizer).
        if 'params' in variables:
            state, params = variables.pop('params')
        else:
            state, params = variables, FrozenDict({})
        del variables  # Delete variables to avoid wasting resources
        # TODO: incorporate state as well in the training
        # return params, state
        return params

    def get_init_param_vec_stacked(self, rng_key: Optional[random.PRNGKey] = None):
        if rng_key is None:
            rng_keys = [self._next_rng_key() for _ in range(self.num_batched_modules)]
        else:
            rng_keys = random.split(rng_key, self.num_batched_modules)
        return vmap(self._init_params_one_model)(jnp.stack(rng_keys))

    @partial(jit, static_argnums=(0,))
    def forward_vec(self, x: jnp.ndarray, param_vectors_stacked: jnp.array):
        assert x.shape[0] == self.num_batched_modules
        return self._forward_vec_vmap(x, param_vectors_stacked)

    @partial(jit, static_argnums=(0,))
    def _apply_one(self, x, params_one):
        return self.base_module.apply({'params': params_one}, x)

    @partial(jit, static_argnums=(0,))
    def _apply_vec(self, xs, params_one):
        return vmap(self._apply_one, in_axes=(0, None))(xs, params_one)

    @partial(jit, static_argnums=(0,))
    def _apply_vec_batch(self, xs, params):
        return vmap(self._apply_vec, in_axes=(None, 0))(xs, params)

    def params_prior(self, weight_prior_std, bias_prior_std) -> tfd.MultivariateNormalDiag:
        prior_stds = []
        params = self._param_vectors_stacked
        list_of_shapes = tree_map(lambda x: x.shape, tree_leaves(params))
        for shape in list_of_shapes:
            if len(shape) == 2:
                prior_stds.append(bias_prior_std * jnp.ones(math.prod(shape[1:])))
            elif len(shape) == 3:
                prior_stds.append(weight_prior_std * jnp.ones(math.prod(shape[1:])))
            else:
                raise ValueError(f'Unknown shape {shape}')

        prior_stds = jnp.concatenate(prior_stds)
        prior_dist = tfd.MultivariateNormalDiag(jnp.zeros_like(prior_stds), prior_stds)
        # assert prior_dist.event_shape == self.batched_model.param_vectors_stacked.shape[-1:]
        return prior_dist

    def __call__(self, x: jnp.ndarray):
        return self._apply_vec_batch(x, self._param_vectors_stacked)


class BatchedModel(BatchedModule, RngKeyMixin):
    def __init__(self, input_size: int, output_size: int, num_batched_modules: int, hidden_layer_sizes=Sequence[int],
                 hidden_activation=Callable, last_activation=Callable, rng_key=jnp.ndarray):
        super().__init__(
            base_module=MLP(hidden_layer_sizes=hidden_layer_sizes, output_size=output_size, input_size=input_size,
                            hidden_activation=hidden_activation, last_activation=last_activation),
            num_batched_modules=num_batched_modules, rng_key=rng_key)

    def forward_vec(self, x: jnp.ndarray, batched_params: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.base_module.input_size
        res = super()._apply_vec_batch(x, batched_params)
        assert res.ndim == 3
        assert res.shape[0] == self.num_batched_modules and res.shape[-1] == self.base_module.output_size
        return res


class MLP(nn.Module):
    hidden_layer_sizes: Sequence[int]
    output_size: int
    input_size: int
    hidden_activation: Callable
    last_activation: Optional[Callable]

    @nn.compact
    def __call__(self, x, train=False):
        for feat in self.hidden_layer_sizes:
            x = nn.Dense(features=feat)(x)
            x = self.hidden_activation(x)
        x = nn.Dense(features=self.output_size)(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x
