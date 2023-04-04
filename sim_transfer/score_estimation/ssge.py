import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Optional, Union

from .abstract import AbstractScoreEstimator, GramMatrixMixin


class SSGE(AbstractScoreEstimator, GramMatrixMixin):
    """
    Implementation of the Spectral Stein Gradient Estimator (https://arxiv.org/abs/1806.02925) which
    estimates the score of a probability distribution based on i.i.d. samples drawn from it.
    """

    def __init__(self, eta: float = 0.1, n_eigen_threshold: float = 0.98, n_eigen_values: Optional[int] = None,
                 bandwidth: Optional[Union[float, jnp.ndarray]] = None,
                 kernel_type: str = 'SE', add_linear_kernel: bool = False):
        """
        Args:
            eta (float): magnitude of identity matrix which to add to kernel matrix to stabilize the eigenvalue
                         decomposition
            n_eigen_threshold (float): minimum percentage of variance explained which is used to choose the number of
                                       eigenvalues / eigenvectors
            n_eigen_values (int, optional): number of leading eigenvalues / eigenvectors to use
            bandwidth (float, optional): kernel bandwidth, if not provided, the bandwidth is chosen with the
                                         median distance heuristic
            kernel_type (str): kernel type, either 'SE' or 'IMQ'
            add_linear_kernel (bool): whether to add a linear kernel to the kernel matrix
        """
        GramMatrixMixin.__init__(self, kernel_type=kernel_type, add_linear_kernel=add_linear_kernel)
        assert 0 < n_eigen_threshold <= 1.0
        self.eta = eta
        self.n_eigen_threshold = n_eigen_threshold
        self.n_eigen_values = n_eigen_values
        self.bandwidth = bandwidth


    def estimate_gradients_s_x(self, queries: jnp.ndarray, samples: jnp.ndarray) -> jnp.ndarray:
        """Estimate the score $\nabla_x log p(x)$ in the queries points given samples from p(x).

        Args:
            queries (jnp.ndarray): queries points for which to estimate the score, array of shape (n_query_points, d)
            samples (jnp.ndarray): i.i.d sample point from p(x), array of shape (n_samples, d)

        Returns:
            Estimated scores, array of shape (n_query_points, d)
        """
        return self.__call__(queries, samples)

    def __call__(self, queries: jnp.ndarray, samples: jnp.ndarray) -> jnp.ndarray:
        """
        x: [..., B, D] sample points
        xm: [..., M, D], index points
        """
        assert queries.shape[-1] == samples.shape[-1]
        m = samples.shape[-2]

        if self.bandwidth is None:
            length_scale = self.bandwith_median_heuristic(samples)
        else:
            length_scale = self.bandwidth

        eigen_values, eigen_vectors, grad_k1 = self._get_eigen_function(samples, length_scale)

        # construct mask to exclude eigenvectors with small eigenvalues
        if self.n_eigen_values is not None:
            n_eigen_mask = jnp.concatenate([jnp.zeros(m - self.n_eigen_values), jnp.ones(self.n_eigen_values)])
        else:
            n_eigen_mask = self._get_n_eigen_mask(eigen_values, self.n_eigen_threshold)
        assert n_eigen_mask.shape == (m,)

        grads = self._get_grads(queries, samples, eigen_vectors, eigen_values, n_eigen_mask, grad_k1, length_scale, m)
        assert grads.shape == queries.shape
        return grads

    def _nystrom_ext(self, x: jnp.ndarray, eval_points: jnp.ndarray, eigen_vectors: jnp.ndarray,
                     eigen_values: jnp.ndarray, bandwidth: Union[float, jnp.ndarray]):
        """
        x: [..., N, D]
        eval_points: [..., M, D]
        eigen_vectors: [..., M, self.n_eigen]
        eigen_values: [..., self.n_eigen]
        returns: [..., N, self.n_eigen], by default n_eigen=m.
        """
        assert eigen_values.shape[-1] == eigen_vectors.shape[-1]
        m = jnp.shape(eval_points)[-2]
        kxm = self.gram(x, eval_points, bandwidth)

        ret = jnp.matmul(kxm, eigen_vectors)
        ret *= (jnp.array(m).astype(ret.dtype) ** 0.5 / jnp.expand_dims(eigen_values, axis=-2))
        return ret

    def _get_eigen_function(self, xm: jnp.ndarray, length_scale: jnp.ndarray):
        """
        xm: [..., M, D]
        m: []
        length_scale: [..., D]
        """
        m = xm.shape[-2]
        kq, grad_k1, grad_k2 = self.grad_gram(xm, xm, length_scale)
        kq += self.eta * jnp.eye(m)
        eigen_values, eigen_vectors = jnp.linalg.eigh(kq)
        return eigen_values, eigen_vectors, grad_k1

    def _get_n_eigen_mask(self, eigen_values: jnp.ndarray, n_eigen_threshold: float) -> jnp.ndarray:
        m = eigen_values.shape[0]
        eigen_arr = jnp.mean(jnp.reshape(eigen_values, [-1, m]), axis=0)
        eigen_arr = jnp.flip(eigen_arr, axis=-1)
        eigen_arr /= jnp.sum(eigen_arr)
        eigen_cum = jnp.cumsum(eigen_arr, axis=-1)
        eigen_less = (jnp.less(eigen_cum, n_eigen_threshold)).astype(jnp.int32)
        return jnp.flip(eigen_less, axis=-1)

    def _get_grads(self, x: jnp.ndarray, xm: jnp.ndarray, eigen_vectors: jnp.ndarray,
                   eigen_values: jnp.ndarray, n_eigen_mask: jnp.ndarray, grad_k1: jnp.ndarray,
                   length_scale: Union[float, jnp.array], m: int) -> jnp.ndarray:
        assert n_eigen_mask.shape == eigen_values.shape
        eigen_ext = self._nystrom_ext(x, xm, eigen_vectors, eigen_values, length_scale)
        grad_k1_avg = jnp.mean(grad_k1, axis=-3)

        beta = - jnp.matmul(jnp.transpose(eigen_vectors), grad_k1_avg)
        beta *= jnp.array(m).astype(beta.dtype) ** 0.5 / (jnp.expand_dims(eigen_values, -1) + 1e-6)
        beta *= jnp.expand_dims(n_eigen_mask, -1)

        grads = jnp.matmul(eigen_ext, beta)
        return grads
