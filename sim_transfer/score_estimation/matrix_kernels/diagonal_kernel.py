from __future__ import division, absolute_import, print_function

from typing import Optional, Tuple

import jax
from jax import numpy as jnp
from .utils import median_heuristic

from sim_transfer.score_estimation.matrix_kernels import DiagonalKernel


class DiagonalGaussian(DiagonalKernel):
    """Diagonal Gaussian kernel."""

    def __init__(self, kernel_hyperparams: Optional[float] = None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_impl(self, x: jnp.ndarray, y: jnp.ndarray, kernel_width: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        d = jnp.shape(x)[-1]
        x_m = jnp.expand_dims(x, -2)  # [M, 1, d]
        y_m = jnp.expand_dims(y, -3)  # [1, N, d]
        diff = x_m - y_m
        dist2 = jnp.sum(diff * diff, -1) # [M, N]
        rbf = jnp.exp(-0.5 * dist2 / kernel_width ** 2) # [M, N]
        divergence = jnp.expand_dims(rbf, -1) * (diff / kernel_width ** 2)

        return rbf, divergence


class DiagonalIMQKernel(DiagonalKernel):
    """Diagonal Inverse Multi-Quadratic (IMQ) kernel."""

    def __init__(self, kernel_hyperparams: Optional[float] = None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_impl(self, x: jnp.ndarray, y: jnp.ndarray, kernel_width: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_m = jnp.expand_dims(x, -2)  # [M, 1, d]
        y_m = jnp.expand_dims(y, -3)  # [1, N, d]
        diff = x_m - y_m
        dist2 = jnp.sum(diff * diff, -1)  # [M, N]
        imq = jax.lax.rsqrt(1 + dist2 / kernel_width ** 2)  # Inverse Multi-Quadratic
        divergence = jnp.expand_dims(imq ** 3, -1) * (diff / kernel_width ** 2)

        return imq, divergence


class DiagonalIMQpKernel(DiagonalKernel):
    """Diagonal Inverse Multi-Quadratic (IMQ) kernel with power parameter."""

    def __init__(self, p: float = 0.5, kernel_hyperparams: Optional[float] = None,
                 heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)
        self._p = p

    def _gram_impl(self, x: jnp.ndarray, y: jnp.ndarray, kernel_width: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_m = jnp.expand_dims(x, -2)  # [M, 1, d]
        y_m = jnp.expand_dims(y, -3)  # [1, N, d]
        diff = x_m - y_m
        dist2 = jnp.sum(diff * diff, -1)  # [M, N]
        imq = 1.0 / (1.0 + dist2 / kernel_width ** 2)
        imq_p = jax.lax.pow(imq, self._p)
        divergence = 2.0 * self._p * jnp.expand_dims(imq * imq_p, -1) * (diff / kernel_width ** 2)
        return imq_p, divergence