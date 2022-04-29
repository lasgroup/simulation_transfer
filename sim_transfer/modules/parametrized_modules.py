import jax
import jax.numpy as jnp

from typing import Callable, Optional, List, Dict, Union
from collections import OrderedDict
from functools import partial, cached_property

from .util import RngKeyMixin


class ParametrizedModule:

    def __init__(self):
        self._param_names = []
        self._vec_to_params_pure = None

    @property
    def params(self) -> Union[List, Dict]:
        return OrderedDict([(name, getattr(self, name)) for name in self._param_names])

    @params.setter
    def params(self, new_params: Union[List, Dict]):
        assert set(new_params.keys()) == set(self._param_names)
        self._set_params(new_params)

    @property
    def param_values(self) -> List[jnp.ndarray]:
        return jax.tree_leaves(self.params)

    @property
    def param_shapes(self) -> List[tuple]:
        return [param.shape for param in self.param_values]

    @property
    def _param_sizes(self) -> List[Union[int, jnp.array]]:
        return [jnp.prod(jnp.array(param_shape)) for param_shape in self.param_shapes]

    @property
    def param_vector(self) -> jnp.ndarray:
        return jnp.concatenate([param.reshape((-1)) for param in self.param_values])

    @param_vector.setter
    def param_vector(self, new_vector: jnp.ndarray):
        assert new_vector.shape == (sum(self._param_sizes),)
        self.params = self._vec_to_params(new_vector)

    @cached_property
    def param_vector_shape(self):
        return self.param_vector.shape

    def init_params(self, inplace: bool = True, rng_key: Optional[jax.random.PRNGKey] = None):
        raise NotImplementedError

    def get_init_param_vec(self, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        init_param_values = jax.tree_leaves(self.init_params(inplace=False, rng_key=rng_key))
        init_vector = jnp.concatenate([param.reshape((-1)) for param in init_param_values])
        assert init_vector.shape == self.param_vector_shape
        return init_vector

    def _register_param(self, name: str, value: jnp.ndarray) -> None:
        assert name not in self._param_names and not hasattr(self, name), (f'Parameter {name} already esists')
        self._param_names.append(name)
        setattr(self, name, value)

    def _set_param(self, name: str, param: jnp.ndarray) -> None:
        assert name in self._param_names and hasattr(self, name)
        setattr(self, name, param)

    def _set_params(self, param_dict: Dict[str, jnp.ndarray]) -> None:
        assert isinstance(param_dict, dict)
        assert set(param_dict.keys()) <= set(self._param_names)
        for param_name, param_value in param_dict.items():
            self._set_param(param_name, param_value)

    def forward(self, x: jnp.ndarray, params: Union[List, Dict]):
        raise NotImplementedError

    def forward_vec(self, x: jnp.ndarray, param_vector: jnp.ndarray):
        return self.forward(x=x, params=self._vec_to_params(param_vector))

    def _vec_to_params(self, vector: jnp.ndarray) -> Union[List, Dict]:
        if self._vec_to_params_pure is None:
            self._create_pure_vec_to_params_fn()
        return self._vec_to_params_pure(vector)

    def _create_pure_vec_to_params_fn(self):
        # creates a pure function for splitting the parameter vector into the params pytree
        split_indices = list(map(int, jnp.cumsum(jnp.array([0, *self._param_sizes]))))
        def _vec_to_params(vector: jnp.ndarray) -> Union[List, Dict]:
            params_split = [vector[...,l_idx:u_idx] for l_idx, u_idx in zip(split_indices[:-1], split_indices[1:])]
            params_dict = OrderedDict([(param_name, flat_param.reshape(param.shape))
                           for flat_param, (param_name, param) in zip(params_split, self.params.items())])
            return params_dict
        self._vec_to_params_pure = _vec_to_params

    def __call__(self, x: jnp.ndarray):
        return self.forward(x, params=self.params)


class Dense(ParametrizedModule, RngKeyMixin):

    def __init__(self, input_size: int, output_size: int, rng_key: jax.random.PRNGKey,
                 activation: Callable = None, bias: bool = True,
                 initializer_w: Callable = jax.nn.initializers.he_uniform(),
                 initializer_b: Callable = jax.nn.initializers.constant(0.01)):
        super().__init__()
        RngKeyMixin.__init__(self, rng_key)

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.initializer_w = initializer_w
        self.initializer_b = initializer_b
        self._register_param(name='w', value=jnp.empty(shape=(self.input_size, self.output_size)))
        if bias:
            self._register_param(name='b', value=jnp.empty(shape=(self.output_size,)))
        self.activation = activation
        self.init_params(inplace=True)

        # creates a pure function for splitting the parameter vector into the params pytree
        self._create_pure_vec_to_params_fn()

    def init_params(self, inplace: bool = True,
                    rng_key: Optional[jax.random.PRNGKey] = None) -> Union[OrderedDict, None]:
        # handle random keys
        if rng_key is None:
            rng_key = self._next_rng_key()
        key_w, key_b = jax.random.split(rng_key, num=2)

        # create dict with parameter initializations
        param_inits = OrderedDict()
        param_inits['w'] = self.initializer_w(key_w, (self.input_size, self.output_size))
        if self.has_bias:
            param_inits['b'] = self.initializer_b(rng_key, (self.output_size,))

        # either return the dict or set the params in place
        if inplace:
            self._set_params(param_inits)
        else:
            return param_inits

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, x: jnp.ndarray, params: Union[List, Dict]) -> jnp.ndarray:
        assert x.shape[-1] == self.input_size
        y = jnp.matmul(x, params['w'])
        if self.has_bias:
            y += params['b']
        if self.activation is not None:
            y = self.activation(y)
        assert y.shape[-1] == self.output_size
        return y

    def __str__(self) -> str:
        return f'Dense({self.input_size}, {self.output_size}, bias={self.has_bias}, activation={self.activation})'


class SequentialModule(ParametrizedModule):

    def __init__(self, list_of_modules: List[Union[ParametrizedModule, Callable]]):
        super().__init__()
        self._modules = list_of_modules
        self._create_pure_vec_to_submodule_vecs_fn()

    @property
    def submodules(self) -> List[Union[ParametrizedModule, Callable]]:
        return self._modules

    @property
    def submodules_parametrized(self) -> List[ParametrizedModule]:
        return list(filter(lambda module: isinstance(module, ParametrizedModule), self.submodules))

    @property
    def num_modules(self) -> int:
        return len(self.submodules)

    @property
    def num_modules_parametrized(self) -> int:
        return len(self.submodules_parametrized)

    @property
    def params(self) -> Union[List, Dict]:
        return [module.params for module in self.submodules_parametrized]

    @params.setter
    def params(self, new_params: Union[List, Dict]):
        assert len(new_params) == self.num_modules_parametrized
        for module, new_params_module in zip(self.submodules_parametrized, new_params):
            module.params = new_params_module

    @property
    def param_shapes(self) -> List[List[tuple]]:
        return [module.param_shapes for module in self.submodules_parametrized]

    @property
    def _param_sizes(self) -> List[List[int]]:
        return [module._param_sizes for module in self.submodules_parametrized]

    @property
    def _submodule_vec_sizes(self) -> List[int]:
        return [sum(module._param_sizes) for module in self.submodules_parametrized]

    @property
    def param_vector(self) -> jnp.ndarray:
        vec = jnp.concatenate([module.param_vector for module in self.submodules_parametrized])
        assert vec.ndim == 1
        return vec

    @param_vector.setter
    def param_vector(self, new_vector: jnp.array):
        for submodule, submodule_vec in zip(self.submodules_parametrized, self._vec_to_submodule_vecs(new_vector)):
            submodule.param_vector = submodule_vec

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, x: jnp.ndarray, params: Union[List, Dict]):
        assert len(params) == self.num_modules_parametrized
        res = x
        param_module_idx = 0
        for module in self.submodules:
            if isinstance(module, ParametrizedModule):
                res = module.forward(res, params[param_module_idx])
                param_module_idx += 1
            else:
                res = module(res)
        assert param_module_idx == self.num_modules_parametrized
        return res

    @partial(jax.jit, static_argnums=(0,))
    def forward_vec(self, x: jnp.ndarray, param_vector: jnp.ndarray):
        res = x
        submodule_vecs = self._vec_to_submodule_vecs(param_vector)
        param_module_idx = 0
        for module in self.submodules:
            if isinstance(module, ParametrizedModule):
                res = module.forward_vec(res, submodule_vecs[param_module_idx])
                param_module_idx += 1
            else:
                res = module(res)
        assert param_module_idx == self.num_modules_parametrized
        return res

    def init_params(self, inplace: bool = True, rng_key: Optional[jax.random.PRNGKey] = None):
        if rng_key is not None:
            rng_keys = jax.random.split(rng_key, self.num_modules)
        else:
            rng_keys = [None] * self.num_modules
        params_init = [module.init_params(inplace=inplace, rng_key=key)
                       for module, key in zip(self.submodules_parametrized, rng_keys)]
        if not inplace:
            return params_init

    def get_init_param_vec(self, rng_key: Optional[jax.random.PRNGKey] = None):
        if rng_key is not None:
            rng_keys = jax.random.split(rng_key, self.num_modules)
        else:
            rng_keys = [None] * self.num_modules
        init_param_vec = jnp.concatenate([module.get_init_param_vec(rng_key=key)
                                          for module, key in zip(self.submodules_parametrized, rng_keys)])
        assert init_param_vec.shape == self.param_vector_shape
        return init_param_vec

    def _vec_to_params(self, vector: jnp.ndarray) -> Union[List, Dict]:
        return [module._vec_to_params(module_vec) for module, module_vec in
                zip(self.submodules_parametrized, self._vec_to_submodule_vecs(vector))]

    def _create_pure_vec_to_submodule_vecs_fn(self):
        # creates a pure function for splitting the parameter vector into of parameters for each submodule
        split_indices = list(map(int, jnp.cumsum(jnp.array([0, *self._submodule_vec_sizes]))))
        def _vec_to_submodule_vecs(vector: jnp.ndarray) -> Union[List, Dict]:
            return [vector[..., l_idx:u_idx] for l_idx, u_idx in zip(split_indices[:-1], split_indices[1:])]
        self._vec_to_submodule_vecs = _vec_to_submodule_vecs

    def __str__(self) -> str:
        module_str = '\n  '.join(map(str, self._modules))
        return f'Sequential[\n  {module_str}\n]'


class MLP(SequentialModule):

    def __init__(self, input_size: int, output_size: int,
                 hidden_layer_sizes: List[int],
                 rng_key: jax.random.PRNGKey,
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None,
                 initializer_w: Callable = jax.nn.initializers.he_uniform(),
                 initializer_b: Callable = jax.nn.initializers.constant(0.01)):
        self.input_size = input_size
        self.output_size = output_size
        sizes = [input_size] + hidden_layer_sizes
        rng_keys = jax.random.split(rng_key, num=len(hidden_layer_sizes)+1)
        layers = [Dense(sizes[i], sizes[i+1], rng_key=rng_keys[i], activation=hidden_activation,
                        initializer_w=initializer_w, initializer_b=initializer_b)
                  for i in range(len(hidden_layer_sizes))]
        layers += [Dense(sizes[-1], output_size, rng_key=rng_keys[-1], activation=last_activation,
                         initializer_w=initializer_w, initializer_b=initializer_b)]
        assert len(layers) == len(hidden_layer_sizes) + 1
        super().__init__(layers)
