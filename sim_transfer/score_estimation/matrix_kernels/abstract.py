import collections
from typing import Optional, Tuple, Callable, Union

from jax import numpy as jnp
from .utils import median_heuristic


class BaseMatrixKernel:
    def __init__(self, kernel_type: str, kernel_hyperparams: Optional[float], heuristic_hyperparams: Callable):
        if kernel_hyperparams is not None:
            heuristic_hyperparams = lambda x, y: kernel_hyperparams
        self._kernel_type = kernel_type
        self._heuristic_hyperparams = heuristic_hyperparams

    def kernel_type(self) -> str:
        return self._kernel_type

    def heuristic_hyperparams(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return self._heuristic_hyperparams(x, y)

    def kernel_operator(self, x: jnp.ndarray, y: jnp.ndarray, kernel_hyperparams: Optional[float], **kwargs):
        pass

    def kernel_matrix(self, x: jnp.ndarray, y: jnp.ndarray, kernel_hyperparams: Optional[float] = None,
                      flatten: bool = True, compute_divergence: bool = True) \
            -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Computes the kernel matrix and optionally the divergence between input arrays x and y.

        Args:
            x: Array of input data points.
            y: Array of input data points.
            kernel_hyperparams: Optional kernel hyperparameters.
            flatten: If True, flattens the output kernel matrix.
            compute_divergence: If True, computes the divergence along with the kernel matrix.

        Returns:
            The kernel matrix and, if compute_divergence is True, the divergence.
        """
        if compute_divergence:
            op, divergence = self.kernel_operator(x, y, compute_divergence=True,
                                                  kernel_hyperparams=kernel_hyperparams)
            return op.kernel_matrix(flatten), divergence
        op = self.kernel_operator(x, y, compute_divergence=False,
                                  kernel_hyperparams=kernel_hyperparams)
        return op.kernel_matrix(flatten)


class SquareCurlFreeKernel(BaseMatrixKernel):
    def __init__(self, kernel_hyperparams: Optional[float] = None, heuristic_hyperparams: Callable = median_heuristic):
        super().__init__('curl-free', kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives(self, x: jnp.ndarray, y: jnp.ndarray, kernel_hyperparams: Optional[float]) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Computes the Gram derivatives for input arrays x and y.

        Args:
            x: Array of input data points.
            y: Array of input data points.
            kernel_hyperparams: Optional kernel hyperparameters.

        Returns:
            A tuple of Gram derivatives.
        """
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams
        x_m = jnp.expand_dims(x, -2)  # [M, 1, d]
        y_m = jnp.expand_dims(y, -3)  # [1, N, d]
        r = x_m - y_m  # [M, N, d]
        norm_rr = jnp.sum(r * r, axis=-1)  # [M, N]
        return self._gram_derivatives_impl(r, norm_rr, kernel_width)

    def _gram_derivatives_impl(self, r: jnp.ndarray, norm_rr: jnp.ndarray, sigma: float) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError('`_gram_derivatives_impl` not implemented.')

    def kernel_energy(self, x: jnp.ndarray, y: jnp.ndarray, kernel_hyperparams: Optional[float] = None,
                      compute_divergence: bool = True) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Computes the kernel energy for input arrays x and y.

        Args:
            x: Array of input data points.
            y: Array of input data points.
            kernel_hyperparams: Optional kernel hyperparameters.
            compute_divergence: If True, computes the divergence along with the kernel energy.

        Returns:
            The kernel energy and, if compute_divergence is True, the divergence.
        """
        d, M, N = jnp.shape(x)[-1], jnp.shape(x)[-2], jnp.shape(y)[-2]
        r, norm_rr, G_1st, G_2nd, _ = self._gram_derivatives(x, y, kernel_hyperparams)

        energy_k = -2. * jnp.expand_dims(G_1st, -1) * r

        if compute_divergence:
            divergence = jnp.array(2 * d).astype(G_1st.dtype) * G_1st \
                         + 4. * norm_rr * G_2nd
            return energy_k, divergence
        return energy_k

    def kernel_operator(self, x: jnp.ndarray, y: jnp.ndarray, kernel_hyperparams: Optional[float] = None,
                        compute_divergence: bool = True, return_matr: bool = False, flatten_matr: bool = True) \
            -> Union[collections.namedtuple, Tuple[collections.namedtuple, jnp.ndarray]]:
        d, M, N = jnp.shape(x)[-1], jnp.shape(x)[-2], jnp.shape(y)[-2] # dimensions
        r, norm_rr, G_1st, G_2nd, G_3rd = self._gram_derivatives(x, y, kernel_hyperparams)
        G_1st = jnp.expand_dims(G_1st, -1)  # [M, N, 1]
        G_2nd = jnp.expand_dims(G_2nd, -1)  # [M, N, 1]

        if compute_divergence:
            coeff = (d + 2.) * G_2nd + 2. * jnp.expand_dims(norm_rr * G_3rd, axis=-1)
            divergence = 4. * coeff * r

        def kernel_op(z: jnp.ndarray) -> jnp.ndarray:
            L = jnp.shape(z)[-1]
            z = jnp.reshape(z, [1, N, d, L])  # [1, N, d, L]
            hat_r = jnp.expand_dims(r, -1)  # [M, N, d, 1]
            dot_rz = jnp.sum(z * hat_r, axis=-2)  # [M, N, L]
            coeff = -4. * G_2nd * dot_rz  # [M, N, L]
            ret = jnp.expand_dims(coeff, -2) * hat_r - 2. * jnp.expand_dims(G_1st, -1) * z
            return jnp.reshape(jnp.sum(ret, axis=-3), [M * d, L])

        def kernel_adjoint_op(z: jnp.ndarray) -> jnp.ndarray:
            L = jnp.shape(z)[-1]
            z = jnp.reshape(z, [M, 1, d, L])  # [M, 1, d, L]
            hat_r = jnp.expand_dims(r, -1)  # [M, N, d, 1]
            dot_rz = jnp.sum(z * hat_r, axis=-2)  # [M, N, L]
            coeff = -4. * G_2nd * dot_rz  # [M, N, L]
            ret = jnp.expand_dims(coeff, -2) * hat_r - 2. * jnp.expand_dims(G_1st, -1) * z
            return jnp.reshape(jnp.sum(ret, axis=-4), [N * d, L])

        def kernel_mat(flatten: bool) -> jnp.ndarray:
            Km = jnp.expand_dims(r, -1) * jnp.expand_dims(r, -2)
            K = -2. * jnp.expand_dims(G_1st, -1) * jnp.eye(d) - 4. * jnp.expand_dims(G_2nd, -1) * Km
            if flatten:
                K = jnp.reshape(jnp.transpose(K, [0, 2, 1, 3]), [M * d, N * d])
            return K

        linear_operator = collections.namedtuple(
            "KernelOperator", ["shape", "dtype", "apply", "apply_adjoint", "kernel_matrix"])

        op = linear_operator(
            shape=[M * d, N * d],
            dtype=x.dtype,
            apply=kernel_op,
            apply_adjoint=kernel_adjoint_op,
            kernel_matrix=kernel_mat,
        )

        if not return_matr:
            if compute_divergence:
                return op, divergence
            return op
        else:
            return op.kernel_matrix(flatten_matr)


class DiagonalKernel(BaseMatrixKernel):
    """Diagonal kernel."""

    def __init__(self, kernel_hyperparams: Optional[float] = None, heuristic_hyperparams=median_heuristic):
        super().__init__('diagonal', kernel_hyperparams, heuristic_hyperparams)

    def _gram(self, x: jnp.ndarray, y: jnp.ndarray, kernel_hyperparams: Optional[float]) \
            -> Tuple[jnp.ndarray, jnp.ndarray]:
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams
        return self._gram_impl(x, y, kernel_width)

    def _gram_impl(self, x: jnp.ndarray, y: jnp.ndarray, kernel_hyperparams: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError('Gram matrix not implemented!')

    def kernel_operator(self, x: jnp.ndarray, y: jnp.ndarray, kernel_hyperparams: Optional[float] = None,
                        compute_divergence: bool = True) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        d = jnp.shape(x)[-1]
        M = jnp.shape(x)[-2]
        N = jnp.shape(y)[-2]
        K, divergence = self._gram(x, y, kernel_hyperparams)

        def kernel_op(z: jnp.ndarray) -> jnp.ndarray:
            # z: [N * d, L]
            L = jnp.shape(z)[-1]
            z = jnp.reshape(z, [N, d * L])
            ret = jnp.matmul(K, z)
            return jnp.reshape(ret, [M * d, L])

        def kernel_adjoint_op(z: jnp.ndarray) -> jnp.ndarray:
            # z: [M * d, L]
            L = jnp.shape(z)[-1]
            z = jnp.reshape(z, [M, d * L])
            ret = jnp.matmul(K, z, transpose_a=True)
            return jnp.reshape(ret, [N * d, L])

        def kernel_mat(flatten: bool) -> jnp.ndarray:
            if flatten:
                return K
            return jnp.expand_dims(jnp.expand_dims(K, -1), -1) * jnp.eye(d)

        linear_operator = collections.namedtuple(
            "Operator", ["shape", "dtype", "apply", "apply_adjoint", "kernel_matrix"])

        op = linear_operator(
            shape=[M * d, N * d],
            dtype=x.dtype,
            apply=kernel_op,
            apply_adjoint=kernel_adjoint_op,
            kernel_matrix=kernel_mat,
        )

        if compute_divergence:
            return op, divergence
        return op