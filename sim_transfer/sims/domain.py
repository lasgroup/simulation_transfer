from typing import Union, Tuple

import jax
import jax.numpy as jnp

class Domain:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims

    def sample_uniformly(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple],
                         support_mode: str = 'full') -> jnp.ndarray:
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

    def sample_uniformly(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple],
                         support_mode: str = 'full') -> jnp.ndarray:
        assert support_mode in ['full', 'partial']
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        if support_mode == 'full':
            sample = jax.random.uniform(key, shape=sample_shape + (self.num_dims,),
                                        minval=self.lower, maxval=self.upper)
        elif support_mode == 'partial':
            assert len(sample_shape) == 1, 'in partial support mode, sample_shape must a tuple of length 1'
            num_samples = sample_shape[0]
            mid = (self.l + self.u) / 2
            # cut of left 10% and right 10% of the domain
            cutoff_scalar = (1 - 0.1)**(1/self.num_dims)
            l = mid + cutoff_scalar * (self.l - mid)
            u = mid + cutoff_scalar * (self.u - mid)

            # exclude the middle 10 % of the domain
            sample = jnp.empty((0, self.num_dims))
            while sample.shape[0] < num_samples:
                key, subkey = jax.random.split(key)
                proposal_points = jax.random.uniform(subkey, shape=(2 * num_samples,) + (self.num_dims,),
                                                     minval=l, maxval=u)
                mask = jnp.logical_not((jnp.linalg.norm((proposal_points - mid) / (self.u - self.l),
                                                        ord=jnp.inf, axis=-1)
                                        < 1 - (1 - 0.05)**self.num_dims))
                sample = jnp.concatenate([sample, proposal_points[mask]], axis=0)
            sample = sample[:num_samples]
        else:
            raise ValueError(f'Unknown support mode {support_mode}')

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
