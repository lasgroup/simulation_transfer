from typing import Union, Tuple, List

import jax
import jax.numpy as jnp


class Domain:
    def __init__(self, num_dims: int):
        self._num_dims = num_dims

    def sample_uniformly(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple],
                         support_mode: str = 'full') -> jnp.ndarray:
        raise NotImplementedError

    @property
    def num_dims(self) -> int:
        return self._num_dims


class HypercubeDomain(Domain):

    def __init__(self, lower: jnp.ndarray, upper: jnp.ndarray):
        assert lower.shape == upper.shape and lower.ndim == upper.ndim == 1
        assert jnp.all(lower <= upper), 'lower bound of domain must be smaller than upper bound'
        num_dims = lower.shape[0]
        super().__init__(num_dims)
        self._lower = lower
        self._upper = upper

    def sample_uniformly(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple],
                         support_mode: str = 'full') -> jnp.ndarray:
        assert support_mode in ['full', 'partial']
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        if support_mode == 'full':
            sample = jax.random.uniform(key, shape=sample_shape + (self._num_dims,),
                                        minval=self._lower, maxval=self._upper)
        elif support_mode == 'partial':
            assert len(sample_shape) == 1, 'in partial support mode, sample_shape must a tuple of length 1'
            num_samples = sample_shape[0]
            mid = (self._lower + self._upper) / 2
            # cut of left 10% and right 10% of the domain
            cutoff_scalar = (1 - 0.1) ** (1 / self._num_dims)
            l = mid + cutoff_scalar * (self._lower - mid)
            u = mid + cutoff_scalar * (self._upper - mid)

            # exclude the middle 10 % of the domain
            sample = jnp.empty((0, self._num_dims))
            while sample.shape[0] < num_samples:
                key, subkey = jax.random.split(key)
                proposal_points = jax.random.uniform(subkey, shape=(2 * num_samples,) + (self._num_dims,),
                                                     minval=l, maxval=u)
                mask = jnp.logical_not((jnp.linalg.norm((proposal_points - mid) / (self._upper - self._lower),
                                                        ord=jnp.inf, axis=-1)
                                        < 1 - (1 - 0.05) ** self._num_dims))
                sample = jnp.concatenate([sample, proposal_points[mask]], axis=0)
            sample = sample[:num_samples]
        else:
            raise ValueError(f'Unknown support mode {support_mode}')

        assert sample.shape[-1] == self._num_dims
        return sample

    @property
    def l(self) -> jnp.ndarray:
        return self._lower

    @property
    def u(self) -> jnp.ndarray:
        return self._upper


class HypercubeDomainWithAngles(HypercubeDomain):

    def __init__(self, angle_indices: Union[List[int], int], lower: jnp.ndarray, upper: jnp.ndarray):
        super().__init__(lower, upper)
        self.angle_indices = [angle_indices] if isinstance(angle_indices, int) else angle_indices
        assert all([i < len(lower) for i in self.angle_indices])
        assert all([lower[i] >= -jnp.pi for i in self.angle_indices])
        assert all([upper[i] <= jnp.pi for i in self.angle_indices])

    def sample_uniformly(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple],
                         support_mode: str = 'full') -> jnp.array:
        samples = super().sample_uniformly(key=key, sample_shape=sample_shape, support_mode=support_mode)
        dimension_arrays = []
        for i in range(samples.shape[-1]):
            if i in self.angle_indices:
                # encode angles with sin and cosine
                dimension_arrays.append(jnp.sin(samples[..., i]))
                dimension_arrays.append(jnp.cos(samples[..., i]))
            else:
                dimension_arrays.append(samples[..., i])
        samples_with_encoded_angles = jnp.stack(dimension_arrays, axis=-1)
        assert samples_with_encoded_angles.shape == samples.shape[:-1] + (self.num_dims,)
        return samples_with_encoded_angles

    @property
    def num_dims(self) -> int:
        return super().num_dims + len(self.angle_indices)

    @property
    def l(self) -> jnp.ndarray:
        lower_bounds = []
        for i in range(self._lower.shape[-1]):
            if i in self.angle_indices:
                lower_bounds.extend([-1., -1.])
            else:
                lower_bounds.append(self._lower[i])
        assert len(lower_bounds) == self.num_dims
        return jnp.array(lower_bounds)

    @property
    def u(self) -> jnp.ndarray:
        upper_bounds = []
        for i in range(self._lower.shape[-1]):
            if i in self.angle_indices:
                upper_bounds.extend([1., 1.])
            else:
                upper_bounds.append(self._upper[i])
        assert len(upper_bounds) == self.num_dims
        return jnp.array(upper_bounds)
