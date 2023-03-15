from typing import Union, Callable

from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp


""" Statistical distance measures """

def mmd2(x: jnp.ndarray, y: jnp.ndarray,
         kernel: tfp.math.psd_kernels.PositiveSemidefiniteKernel,
         include_diag: bool = True) -> Union[jnp.ndarray, float]:
    """ Computes the MMD^2 between two samples x and y """
    assert x.ndim == y.ndim >= 2, 'x and y must be at least 2-dimensional'
    assert x.shape[-1] == y.shape[-1], 'x and y must have the same diimensionality'
    n, m = x.shape[-2], y.shape[-2]
    Kxx = kernel.matrix(x, x)
    Kyy = kernel.matrix(y, y)
    Kxy = kernel.matrix(x, y)
    if include_diag:
        mmd = jnp.mean(Kxx, axis=(-2, -1)) + jnp.mean(Kyy, axis=(-2, -1)) - 2 * jnp.mean(Kxy, axis=(-2, -1))
    else:
        mmd = jnp.sum(Kxx * (1 - jnp.eye(n)), axis=(-2, -1)) / (n * (n-1)) \
              + jnp.sum(Kyy * (1 - jnp.eye(m)), axis=(-2, -1)) / (m * (m-1)) \
              - 2 * jnp.sum(Kxy, axis=(-2, -1)) / (n * m)
    return mmd


""" Calibration metrics """
def _get_mean_std_from_dist(dist: tfp.distributions.Distribution):
    if isinstance(dist.mean, jnp.ndarray):
        return dist.mean, dist.stddev
    elif isinstance(dist.mean, Callable):
        return dist.mean(), dist.stddev()
    else:
        raise RuntimeError('dist.mean must be a callable or a jnp.ndarray')

def _check_dist_and_sample_shapes(dist: tfp.distributions.Distribution, y: jnp.ndarray) -> None:
    if y.ndim == 1:
        assert dist.batch_shape == y.shape, f'dist.batch_shape {dist.batch_shape } must be equal to y.shape {y.shape}'
    elif y.ndim == 2:
        assert (dist.batch_shape + dist.event_shape) == y.shape or \
               (dist.batch_shape == y.shape and len(dist.event_shape) == 0), \
            'dist.batch_shape must be equal to y.shape[0]'
    else:
        raise NotImplementedError('y must be 1- or 2-dimensional')

def calibration_error_cum(dist: tfp.distributions.Distribution, y: jnp.ndarray) -> Union[jnp.ndarray, float]:
    """ Computes the calibration error of a distribution w.r.t. a sample based the differences in
        empirical cdf and expected cdf. """
    _check_dist_and_sample_shapes(dist, y)
    mean, std = _get_mean_std_from_dist(dist)
    res2 = ((mean - y) / std)**2
    conf_values = jnp.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.977, 0.99])
    emp_freqs = jnp.mean(res2[..., None] <= tfp.distributions.Chi2(df=1).quantile(conf_values), axis=0)
    return jnp.mean(jnp.abs(emp_freqs - conf_values))


def calibration_error_bin(dist: tfp.distributions.Distribution, y: jnp.ndarray) -> Union[jnp.ndarray, float]:
    """ Computes the calibration error of a distribution w.r.t. a sample based on 10 % bins of the probability mass """
    _check_dist_and_sample_shapes(dist, y)
    mean, std = _get_mean_std_from_dist(dist)
    res2 = ((mean - y) / std)**2
    conf_values = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    n_bins = len(conf_values) - 1
    quantiles = tfp.distributions.Chi2(df=1).quantile(conf_values)
    emp_freqs = jnp.mean((quantiles[:-1] <= res2[..., None]) & (res2[..., None] <= quantiles[1:]), axis=0)
    expected_freqs = conf_values[1:] - conf_values[:-1]
    return jnp.mean(jnp.sum(jnp.abs(emp_freqs - expected_freqs), axis=-1)) / (2 - 2 / n_bins)


if __name__ == '__main__':
    import jax.numpy as jnp
    import jax

    key1, key2 = jax.random.split(jax.random.PRNGKey(45645), 2)

    x = jax.random.uniform(key1, shape=(20000, 1), minval=-5, maxval=5)
    y_mean = 5 * x + 2

    dist = tfp.distributions.Normal(loc=y_mean, scale=1)
    y = dist.sample(seed=key2)

    dist_test = tfp.distributions.Normal(loc=y_mean, scale=1.)
    print(calibration_error_cum(dist_test, y))
    print(calibration_error_bin(dist_test, y))

    dist_test = tfp.distributions.Normal(loc=y_mean, scale=2.0)
    print(calibration_error_cum(dist_test, y))
    print(calibration_error_bin(dist_test, y))