import jax.lax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.math.psd_kernels.internal.util import pairwise_square_distance_matrix

from typing import Union, Tuple, Optional


class AbstractScoreEstimator:

    def __init__(self, add_linear_kernel: bool = False):
        self.add_linear_kernel = add_linear_kernel

    def estimate_gradients_s_x(self, queries: jnp.ndarray, samples: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def estimate_gradients_s(self, x: jnp.ndarray) -> jnp.ndarray:
        """Estimate the score $\nabla_x log p(x)$ in the sample points from p(x).

        Args:
            x (jnp.ndarray): i.i.d sample point from p(x), array of shape (n_samples, d)

        Returns:
            Estimated scores, array of shape (n_samples, d)
        """
        return self.estimate_gradients_s_x(queries=x, samples=x)


class GramMatrixMixin:

    def __init__(self, kernel_type: str = 'SE', add_linear_kernel: bool = False):
        assert kernel_type in ['se', 'imq', 'SE', 'IMQ']
        self.kernel_type = kernel_type
        self.add_linear_kernel = add_linear_kernel

    @staticmethod
    def se_fn(r2: jnp.array) -> jnp.array:
        return jnp.exp(-r2 / 2)

    @staticmethod
    def img_fn(r2: jnp.array) -> jnp.array:
        return jax.lax.rsqrt(1 + r2)

    def gram(self, x1: jnp.ndarray, x2: jnp.ndarray, bandwidth: Union[float, jnp.ndarray],
             add_linear_kernel: Optional[bool] = None) -> jnp.ndarray:
        """
        x1: [..., N, D]
        x2: [..., M, D]
        bandwidth: [..., D]
        returns: [..., N, M]
        """
        r2 = pairwise_square_distance_matrix(x1/bandwidth, x2/bandwidth, feature_ndims=1)

        if self.kernel_type in ['se', 'SE']:
            K = self.se_fn(r2)
        else:
            K = self.img_fn(r2)
        if (self.add_linear_kernel if add_linear_kernel is None else add_linear_kernel):
            K += jnp.matmul(x1, x2.T)
        assert K.shape == (x1.shape[-2], x2.shape[-2])
        return K

    def grad_gram(self, x1: jnp.ndarray, x2: jnp.ndarray,
                  bandwidth: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        x1: [..., N, D] # these are the samples usually
        x2: [..., M, D]  # these are the samples usually
        bandwidth: [..., D]
        returns: [..., N, M], [..., N, M, D], [..., N, M, D]
        """
        n, m, d = x1.shape[-2], x2.shape[-2], x1.shape[-1]
        assert d == x1.shape[-1] == x2.shape[-1]
        x1_expanded = jnp.expand_dims(x1, -2)
        x2_expanded = jnp.expand_dims(x2, -3)
        diff = x1_expanded - x2_expanded

        if self.kernel_type in ['se', 'SE']:
            kxx = self.se_fn(jnp.sum((diff/bandwidth) ** 2, axis=-1))
            kxx_grad = kxx
        elif self.kernel_type in ['imq', 'IMQ']:
            r2 = jnp.sum((diff / bandwidth) ** 2, axis=-1)
            kxx = self.img_fn(r2)
            kxx_grad = (1 + r2) ** (-3 / 2)

        diff_scaled = diff / bandwidth**2
        grad_1 = jnp.expand_dims(kxx_grad, -1) * (-diff_scaled)
        grad_2 = jnp.expand_dims(kxx_grad, -1) * diff_scaled

        if self.add_linear_kernel:
            grad_1 = grad_1 + jnp.repeat(x2_expanded, n, axis=-3)
            grad_2 = grad_2 + jnp.repeat(x1_expanded, m, axis=-2)
            kxx += jnp.matmul(x1, x2.T)
        return kxx, grad_1, grad_2

    @staticmethod
    def bandwith_median_heuristic(samples: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the median distance between pairs of points in a given dataset using a pairwise distance matrix.
        This heuristic is often used to estimate the bandwidth parameter in kernel density estimation
        and other non-parametric methods.

        Args:
            samples (jnp.ndarray): A 2D array-like object (N x M) containing the data points,
            where N is the number of samples and M is the number of features.

        Returns:
            jnp.ndarray: A scalar value representing the median distance between pairs of points in the dataset.
        """
        pdist_mat = jnp.sqrt(pairwise_square_distance_matrix(samples, samples, feature_ndims=1))  # [N x M]
        median_distance = jnp.median(jnp.ravel(pdist_mat))
        return median_distance

