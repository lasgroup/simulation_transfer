from typing import Union, Tuple

import jax
import jax.numpy as jnp

class Domain:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims

    def sample_uniformly(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple]):
        raise NotImplementedError

    def project_to_domain(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class HypercubeDomain(Domain):

    def __init__(self, lower: jnp.ndarray, upper: jnp.ndarray):
        assert lower.shape == upper.shape and lower.ndim == upper.ndim == 1
        assert jnp.all(lower <= upper), 'lower bound of domain must be smaller than upper bound'
        num_dims = lower.shape[0]
        super().__init__(num_dims)
        self.lower = lower
        self.upper = upper

    def sample_uniformly(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple]):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        sample = jax.random.uniform(key, shape=sample_shape + (self.num_dims,), minval=self.lower, maxval=self.upper)
        assert sample.shape[-1] == self.num_dims
        return sample

    def project_to_domain(self, x: jnp.ndarray) -> jnp.ndarray:
        res = jnp.clip(x, self.lower, self.upper)
        assert res.shape == x.shape
        return res

    @property
    def l(self) -> jnp.ndarray:
        return self.lower

    @property
    def u(self) -> jnp.ndarray:
        return self.upper
