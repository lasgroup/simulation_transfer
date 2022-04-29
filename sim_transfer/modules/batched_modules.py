import jax
import jax.numpy as jnp

from typing import  Optional, List, Dict, Union
from functools import partial, cached_property

from .parametrized_modules import MLP, ParametrizedModule
from .util import RngKeyMixin, tree_stack, tree_unstack


class BatchedModuleMixin(RngKeyMixin):

    def __init__(self, base_module: ParametrizedModule, num_batched_modules: int,
                 rng_key: jax.random.PRNGKey):
        super().__init__(rng_key)
        self.base_module = base_module
        self.num_batched_modules = num_batched_modules

        # initialize the batched module
        self._param_vectors_stacked = self.get_init_param_vec_stacked()

        # vmap
        self._forward_vec_vmap = jax.vmap(self.base_module.forward_vec)

    def get_init_param_vec_stacked(self, rng_key: Optional[jax.random.PRNGKey] = None):
        if rng_key is None:
            rng_keys = [self._next_rng_key() for _ in range(self.num_batched_modules)]
        else:
            rng_keys = jax.random.split(rng_key, self.num_batched_modules)
        init_vec_stacked = jnp.stack([self.base_module.get_init_param_vec(rng_key=key)
         for key in rng_keys], axis=0)
        assert init_vec_stacked.shape == (self.num_batched_modules, self.module_vector_size)
        return init_vec_stacked

    @cached_property
    def module_vector_size(self) -> int:
        assert len(self.base_module.param_vector_shape) == 1
        return int(self.base_module.param_vector_shape[0])

    @property
    def params_stacked(self) -> Union[List, Dict]:
        # stacks each parameter along the first dimension and returns them in the same pytree structure as the
        # base model
        return self._stacked_vecs_to_stacked_params(self.param_vectors_stacked)

    @params_stacked.setter
    def params_stacked(self, params_stacked: Union[List, Dict]):
        pytree_list = tree_unstack(params_stacked)
        assert len(pytree_list) == self.num_batched_modules
        param_vecs_stacked = jnp.stack([jnp.concatenate(list(map(lambda p: p.reshape((-1,)), jax.tree_leaves(p_tree))))
                                        for p_tree in pytree_list])
        self.param_vectors_stacked = param_vecs_stacked

    @property
    def param_vectors_stacked(self) -> jnp.ndarray:
        return self._param_vectors_stacked

    @param_vectors_stacked.setter
    def param_vectors_stacked(self, new_vecs_stacked: jnp.ndarray):
        assert new_vecs_stacked.shape == (self.num_batched_modules, self.module_vector_size)
        self._param_vectors_stacked = new_vecs_stacked

    def forward(self, x: jnp.ndarray, params: Union[List, Dict]):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def forward_vec(self, x: jnp.ndarray, param_vectors_stacked: jnp.array):
        assert x.shape[0] == self.num_batched_modules
        return self._forward_vec_vmap(x, param_vectors_stacked)

    def _stacked_vecs_to_stacked_params(self, stacked_vecs: jnp.ndarray) -> Union[List, Dict]:
        assert stacked_vecs.shape == (self.num_batched_modules, self.module_vector_size)
        params_each = [self.base_module._vec_to_params(vec) for vec in stacked_vecs]
        return tree_stack(params_each)

    def __call__(self, x: jnp.ndarray):
        return self.forward_vec(x, param_vectors_stacked=self.param_vectors_stacked)


class BatchedMLP(BatchedModuleMixin, RngKeyMixin):

    def __init__(self, *args, rng_key: jax.random.PRNGKey, num_batched_modules: int = 10, **kwargs):
        rng_key_mlp, rng_self = jax.random.split(rng_key, 2)
        super().__init__(base_module=MLP(*args, rng_key=rng_key_mlp, **kwargs), num_batched_modules=num_batched_modules,
                         rng_key=rng_self)

    def forward_vec(self, x: jnp.ndarray, param_vectors_stacked: jnp.array) -> jnp.array:
        if x.ndim == 2:
            # tile the inputs if they are 2-dimensional
            x = jnp.repeat(x[None, :, :], repeats=self.num_batched_modules, axis=0)
        assert x.ndim == 3 and x.shape[-1] == self.base_module.input_size
        res = super().forward_vec(x, param_vectors_stacked)
        assert res.ndim == 3
        assert res.shape[0] == self.num_batched_modules and res.shape[-1] == self.base_module.output_size
        return res
