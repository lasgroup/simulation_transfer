import jax
import jax.numpy as jnp

from typing import Optional, Callable, Union, Tuple, List
from sim_transfer.score_estimation.matrix_kernels import CurlFreeIMQKernel, CurlFreeSEKernel
from sim_transfer.score_estimation.abstract import AbstractScoreEstimator


class NuMethod(AbstractScoreEstimator):

    def __init__(self,
                 lam: Optional[float] = 1e-4,
                 num_iter: Optional[int] = None,
                 kernel_type: str = 'curl_free_imq',
                 bandwidth: float = 1.0,
                 nu: float = 1.0):
        super().__init__()

        if lam is not None and num_iter is not None:
            raise ValueError('Cannot specify `lam` and `iternum` simultaneously.')
        if lam is None and num_iter is None:
            raise ValueError('Both `lam` and `iternum` are `None`.')
        if num_iter is not None:
            lam = 1.0 / (num_iter ** 2)
        else:
            num_iter = jnp.array(1.0 / jnp.sqrt(lam)).astype(jnp.int32) + 1

        if kernel_type == 'curl_free_imq':
            self._kernel = CurlFreeIMQKernel(kernel_hyperparams=bandwidth)
        elif kernel_type == 'curl_free_se':
            self._kernel = CurlFreeSEKernel(kernel_hyperparams=bandwidth)
        else:
            raise NotImplementedError(f'Kernel type {kernel_type} is not implemented.')

        self._lam = lam
        self._nu = nu
        self.num_iter = num_iter
        self.bandwidth = bandwidth
        self.name = "Nu_method"

    def estimate_gradients_s_x(self, queries: jnp.ndarray, samples: jnp.ndarray) -> jnp.array:
        a, c = self.fit(samples)
        score_estimate = self._compute_gradients(queries=queries, samples=samples, a=a, c=c)
        assert queries.shape == score_estimate.shape
        return score_estimate

    def fit(self, samples: jnp.array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Compute the nu-iteration coefficient a and vector c through fixed point iteration."""
        if self.bandwidth is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        else:
            kernel_hyperparams = self.bandwidth

        self._kernel_hyperparams = kernel_hyperparams

        # dimension sizes
        M, d = jnp.shape(samples)[-2], jnp.shape(samples)[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples, kernel_hyperparams=kernel_hyperparams)

        # H_dh: [Md, 1]
        H_dh = jnp.reshape(jnp.mean(K_div, axis=-2), (M * d, 1))

        def get_next(packed):
            t, a, pa, c, pc = packed # nc <- c <- pc
            u, w = self._nu_method_coef_polynomials(t)
            nc = (1. + u) * c - w * (a * H_dh + K_op.apply(c)) / M - u * pc
            na = (1. + u) * a - u * pa - w
            packed_ret = [t + 1, na, a, nc, c]
            return packed_ret

        a1 = -(4. * self._nu + 2) / (4. * self._nu + 1)
        ret = jax.lax.while_loop(
            cond_fun=lambda packed: packed[0] <= self.num_iter,
            body_fun=get_next,
            init_val=[2, a1, 0., jnp.zeros_like(H_dh), jnp.zeros_like(H_dh)]
        )
        a, c = ret[1], ret[3]
        return a, c

    def _nu_method_coef_polynomials(self, t: Union[jnp.array, float, int]) -> \
            Tuple[Union[jnp.array, float, int], Union[jnp.array, float, int]]:
        nu = self._nu
        u = (t - 1.) * (2. * t - 3.) * (2. * t + 2. * nu - 1.) \
            / ((t + 2. * nu - 1.) * (2. * t + 4. * nu - 1.) * (2. * t + 2. * nu - 3.))
        w = 4. * (2. * t + 2. * nu - 1.) * (t + nu - 1.) / ((t + 2. * nu - 1.) * (2. * t + 4. * nu - 1.))
        return u, w

    def _compute_energy(self, queries: jnp.array, samples: jnp.array,
                        coeff: Tuple[jnp.array, jnp.array]) -> jnp.array:
        Kxq, div_xq = self._kernel.kernel_energy(queries, samples,
                                                 kernel_hyperparams=self._kernel_hyperparams)
        Kxq = jnp.reshape(Kxq, (jnp.shape(queries)[-2], -1))
        div_xq = jnp.mean(div_xq, axis=-1) * coeff[0]
        energy = jnp.reshape(jnp.matmul(Kxq, coeff[1]), [-1]) + div_xq
        return energy

    def _compute_gradients(self, queries: jnp.array, samples: jnp.array,
                           a: jnp.array, c: jnp.array) -> jnp.array:
        d = jnp.shape(queries)[-1]
        Kxq_op, div_xq = self._kernel.kernel_operator(queries, samples, kernel_hyperparams=self._kernel_hyperparams)
        div_xq = jnp.mean(div_xq, axis=-2) * a
        grads = Kxq_op.apply(c)
        grads = jnp.reshape(grads, (-1, d)) + div_xq  # same
        return grads
