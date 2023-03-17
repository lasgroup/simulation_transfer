from typing import Optional, Union
from tensorflow_probability.substrates.jax.math.psd_kernels.internal.util import pairwise_square_distance_matrix

import jax.numpy as jnp
import jax

class KDE:
    """ Simple Kernel Density Estimator """

    def __init__(self, bandwidth: Optional[float] = None):
        self.bandwidth = bandwidth

    def estimate_gradients_s_x(self, query: jnp.ndarray, samples: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda query: jnp.sum(self.__call__(query, samples)))(query)

    def density_estimates_log_prob(self, query: jnp.array, samples: jnp.array) -> jnp.array:
        return self.__call__(query, samples)

    @staticmethod
    def _pairwise_distance(x: jnp.array, y: jnp.array, bandwidth: Union[jnp.array, float] = 1.) -> jnp.array:
        n, m = x.shape[-2], y.shape[-2]
        assert x.shape[-1] == y.shape[-1]
        dist_mat = jnp.sum(((x[..., :, None, :] - y[..., None, :, :]) / bandwidth) ** 2, axis=-1)
        assert dist_mat.shape[-2] == n and dist_mat.shape[-1] == m
        return dist_mat

    def __call__(self, query: jnp.array, samples: jnp.array, expected_std: Optional[float] = None):
        assert query.shape[-1] == samples.shape[-1], 'queries and samples must have the same dimensionality'
        n, d = samples.shape[0], samples.shape[1]
        if self.bandwidth == None:
            std = jnp.std(samples, axis=0) if expected_std is None else expected_std
            bandwidth = (std + 1e-3) * n ** (-1. / (d + 4))  # Scott's rule of thumb
        else:
            bandwidth = self.bandwidth
        if isinstance(bandwidth, float) or (not bandwidth.shape == (d, )):
            bandwidth *= jnp.ones(d)
        dists = self._pairwise_distance(query, samples, bandwidth=bandwidth)
        prior_logprob = jax.scipy.special.logsumexp(-dists/2., axis=1) - \
                        (jnp.log(n) + 0.5 * jnp.sum(jnp.log(bandwidth)) + 0.5 * d * jnp.log(2 * jnp.pi))
        return prior_logprob