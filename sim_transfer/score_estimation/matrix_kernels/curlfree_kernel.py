from typing import Optional, Union, Tuple
from jax import numpy as jnp
import jax


from . import SquareCurlFreeKernel
from .utils import median_heuristic


class CurlFreeIMQKernel(SquareCurlFreeKernel):
    """Inverse Multi-Quadratic curl-free kernel."""

    def __init__(self, kernel_hyperparams: Optional[float] = None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives_impl(self, r: jnp.ndarray, norm_rr: jnp.ndarray, sigma: float) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # computes the first three derivatives of isotropic kernel function
        inv_sqr_sigma = 1.0 / jnp.square(sigma)
        imq = jax.lax.rsqrt(1.0 + norm_rr * inv_sqr_sigma)  # [M, N]
        imq_2 = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        G_1st = -0.5 * imq_2 * inv_sqr_sigma * imq
        G_2nd = -1.5 * imq_2 * inv_sqr_sigma * G_1st
        G_3rd = -2.5 * imq_2 * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd


class CurlFreeIMQpKernel(SquareCurlFreeKernel):
    def __init__(self, p: float = 0.5, kernel_hyperparams: Optional[float] = None,
                 heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)
        self._p = p

    def _gram_derivatives_impl(self, r: jnp.ndarray, norm_rr: jnp.ndarray, sigma: float) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # computes the first three derivatives of isotropic kernel function
        inv_sqr_sigma = 1.0 / jnp.square(sigma)
        imq = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        imq_p = jax.lax.pow(imq, self._p)  # [M, N]
        G_1st = -(0. + self._p) * imq * inv_sqr_sigma * imq_p
        G_2nd = -(1. + self._p) * imq * inv_sqr_sigma * G_1st
        G_3rd = -(2. + self._p) * imq * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd


class CurlFreeSEKernel(SquareCurlFreeKernel):
    """Curl-free Squared Exponential kernel."""

    def __init__(self, kernel_hyperparams: Optional[float] = None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives_impl(self, r: jnp.ndarray, norm_rr: jnp.ndarray, sigma: float) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # computes the first three derivatives of isotropic kernel function
        inv_sqr_sigma = 0.5 / jnp.square(sigma)
        rbf = jnp.exp(-norm_rr * inv_sqr_sigma)
        G_1st = -rbf * inv_sqr_sigma
        G_2nd = -G_1st * inv_sqr_sigma
        G_3rd = -G_2nd * inv_sqr_sigma
        return r, norm_rr, G_1st, G_2nd, G_3rd

