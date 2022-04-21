import jax.numpy as jnp
from jax import jit
from functools import partial

from typing import Union, Tuple, Optional

class AbstractScoreEstimator:

    def __init__(self, add_linear_kernel: bool = False):
        self.add_linear_kernel = add_linear_kernel

    @staticmethod
    def rbf_kernel(x1: jnp.ndarray, x2: jnp.ndarray, bandwidth: Union[float, jnp.ndarray]):
        return jnp.exp(-jnp.sum(jnp.square((x1 - x2) / bandwidth), axis=-1) / 2)

    @partial(jit, static_argnums=(0, 4))
    def gram(self, x1: jnp.ndarray, x2: jnp.ndarray, bandwidth: Union[float, jnp.ndarray],
             add_linear_kernel: Optional[bool] = None) -> jnp.ndarray:
        """
        x1: [..., N, D]
        x2: [..., M, D]
        bandwidth: [..., D]
        returns: [..., N, M]
        """
        x_row = jnp.expand_dims(x1, -2)
        x_col = jnp.expand_dims(x2, -3)
        K = self.rbf_kernel(x_row, x_col, bandwidth)
        if (self.add_linear_kernel if add_linear_kernel is None else add_linear_kernel):
            K += jnp.matmul(x1, x2.T)
        assert K.shape == (x1.shape[-2], x2.shape[-2])
        return K

    @partial(jit, static_argnums=0)
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
        kxx = self.gram(x1, x2, bandwidth, add_linear_kernel=False)
        x1_expanded = jnp.expand_dims(x1, -2)
        x2_expanded = jnp.expand_dims(x2, -3)

        diff = (x1_expanded - x2_expanded) / (bandwidth ** 2)
        grad_1 = jnp.expand_dims(kxx, -1) * (-diff)
        grad_2 = jnp.expand_dims(kxx, -1) * diff

        if self.add_linear_kernel:
            grad_1 = grad_1 + jnp.repeat(x2_expanded, n, axis=-3)
            grad_2 = grad_2 + jnp.repeat(x1_expanded, m, axis=-2)
            kxx += jnp.matmul(x1, x2.T)
        return kxx, grad_1, grad_2

    @partial(jit, static_argnums=0)
    def _median_heuristic(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        x: [..., N, D]
        xm: [..., M, D]
        returns: [..., 1, 1, d]
        """
        x1 = jnp.expand_dims(x1, -2)
        x2 = jnp.expand_dims(x2, -3)

        pdist_mat = jnp.sum(jnp.sqrt((x1 - x2) ** 2), axis=-1)  # [N x M]
        median_distance = jnp.median(jnp.ravel(pdist_mat))
        return median_distance
