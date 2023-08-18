import math
from functools import partial
from typing import Optional, Sequence, Callable, Union, List, Dict, Tuple, Any

import flax.linen
import jax.nn
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.core import FrozenDict
from flax.linen.dtypes import promote_dtype
from jax import random, vmap, jit
from jax._src import dtypes
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_leaves

from sim_transfer.modules.util import RngKeyMixin

Array = Any


class BatchedModule(RngKeyMixin):
    def __init__(self, base_module: nn.Module, num_batched_modules: int, rng_key: random.PRNGKey):
        super().__init__(rng_key)
        self.base_module = base_module
        self.num_batched_modules = num_batched_modules

        self._param_vectors_stacked = self.get_init_param_vec_stacked()
        self.flatten_batch, self.unravel_batch = self._get_flatten_batch_fns()

    @property
    def param_vectors_stacked(self) -> FrozenDict:
        return self._param_vectors_stacked

    @param_vectors_stacked.setter
    def param_vectors_stacked(self, params_stacked: Union[List, Dict]):
        self._param_vectors_stacked = params_stacked

    def get_init_param_vec_stacked(self, rng_key: Optional[random.PRNGKey] = None) -> FrozenDict:
        if rng_key is None:
            rng_keys = [self._next_rng_key() for _ in range(self.num_batched_modules)]
        else:
            rng_keys = random.split(rng_key, self.num_batched_modules)
        return vmap(self._init_params_one_model)(jnp.stack(rng_keys))

    @partial(jit, static_argnums=(0,))
    def forward_vec(self, x: jnp.ndarray, param_vectors_stacked: jnp.array):
        assert x.shape[0] == self.num_batched_modules
        return self._forward_vec_vmap(x, param_vectors_stacked)

    def reinit_params(self, rng_key: Optional[random.PRNGKey] = None):
        """ Reinitialize the parameters of the batched model """
        if rng_key is None:
            rng_key = self._next_rng_key()
        self._param_vectors_stacked = self.get_init_param_vec_stacked(rng_key)

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

    def _get_flatten_batch_fns(self) -> Tuple[Callable, Callable]:
        """ Construct function for flattening and unraveling the parameters"""
        one_batch = self._init_params_one_model(random.PRNGKey(0))
        flat_params, unravel_callable = ravel_pytree(one_batch)

        def only_flatten(params):
            return ravel_pytree(params)[0]

        flatten_batch = jit(vmap(only_flatten))
        unravel_batch = jit(vmap(unravel_callable))
        return flatten_batch, unravel_batch

    def _init_params_one_model(self, key: random.PRNGKey) -> FrozenDict:
        variables = self.base_module.init(key, jnp.ones(shape=(self.base_module.input_size,)))
        # TODO: incorporate state as well in the training?
        return variables['params']


class UniformBiasInitializer:
    """ Initializes the biases uniformly in [- 1/sqrt(fan_in), 1/sqrt(fan_in)] """

    def __call__(self, key: jax.random.PRNGKey, shape: Sequence[int],
                 fan_in: int, dtype: Any = None) -> jnp.array:
        dtype = dtypes.canonicalize_dtype(dtype)
        scale = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        return jax.random.uniform(key, shape, dtype) * 2 * scale - scale


class CustomDense(nn.Dense):
    """
    The same as jax.nn.Dense except that it allows to use UniformBiasInitializer
    which also uses fan_in to determine the scale of the uniform bias initialization
    """

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.param('kernel',
                            self.kernel_init,
                            (jnp.shape(inputs)[-1], self.features),
                            self.param_dtype)
        if self.use_bias:
            if isinstance(self.bias_init, UniformBiasInitializer):
                bias = self.param('bias', self.bias_init, (self.features,),
                                  jnp.shape(inputs)[-1], self.param_dtype)
            else:
                bias = self.param('bias', self.bias_init, (self.features,),
                                  self.param_dtype)
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        y = jax.lax.dot_general(inputs, kernel,
                                (((inputs.ndim - 1,), (0,)), ((), ())),
                                precision=self.precision)
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class MLP(nn.Module):
    input_size: int
    output_size: int
    hidden_layer_sizes: Sequence[int]
    hidden_activation: Callable = jax.nn.leaky_relu
    last_activation: Optional[Callable] = None
    kernel_init: Callable[[jax.random.PRNGKey, Tuple[int, ...], Any], Any] = jax.nn.initializers.he_uniform()
    bias_init: Callable[[jax.random.PRNGKey, Tuple[int, ...], Any], Any] = UniformBiasInitializer()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        for feat in self.hidden_layer_sizes:
            x = CustomDense(features=feat, kernel_init=self.kernel_init,
                            bias_init=self.bias_init)(x)
            if self.layer_norm:
                x = flax.linen.LayerNorm(reduction_axes=-1)(x)
            x = self.hidden_activation(x)
        x = CustomDense(features=self.output_size)(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x


class BatchedMLP(BatchedModule, RngKeyMixin):
    def __init__(self, input_size: int, output_size: int, num_batched_modules: int, hidden_layer_sizes: Sequence[int],
                 rng_key: jax.random.PRNGKey, hidden_activation: Callable = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        super().__init__(
            base_module=MLP(hidden_layer_sizes=hidden_layer_sizes, output_size=output_size, input_size=input_size,
                            hidden_activation=hidden_activation, last_activation=last_activation),
            num_batched_modules=num_batched_modules,
            rng_key=rng_key)

    def forward_vec(self, x: jnp.ndarray, batched_params: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.base_module.input_size
        res = super()._apply_vec_batch(x, batched_params)
        assert res.ndim == 3
        assert res.shape[0] == self.num_batched_modules and res.shape[-1] == self.base_module.output_size
        return res
