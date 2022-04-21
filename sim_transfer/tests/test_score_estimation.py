from tensorflow_probability.substrates import jax as tfp
from scipy import spatial
import jax
import jax.numpy as jnp
import unittest
import numpy as np


from sim_transfer.score_estimation import SSGE
from sim_transfer.score_estimation.abstract import AbstractScoreEstimator


dist = tfp.distributions.Normal(loc=jnp.array([0.]), scale=jnp.array([1.0]))
dist = tfp.distributions.Independent(dist, reinterpreted_batch_ndims=1)


class TestAbstractScoreEstimator(unittest.TestCase):

    def setUp(self) -> None:
        key = jax.random.PRNGKey(68)
        key1, key2 = jax.random.split(key)
        self.x1 = jax.random.normal(key1, shape=(3, 2))
        self.x2 = jax.random.normal(key2, shape=(2, 2))

    @staticmethod
    def rbf_kernel(x1, x2, length_scale):
        return jnp.exp(- jnp.linalg.norm((x1 - x2) / length_scale) ** 2 / 2)

    def test_gram_matrix(self) -> None:
        for add_linear_kernel in [True, False]:
            length_scale = 0.4
            score_estimator = AbstractScoreEstimator(add_linear_kernel=add_linear_kernel)

            def kernel(x1, x2):
                k = self.rbf_kernel(x1, x2, length_scale)
                if add_linear_kernel:
                    k += jnp.dot(x1, x2)
                return k

            K = score_estimator.gram(self.x1, self.x2, length_scale)

            for i in range(self.x1.shape[0]):
                for j in range(self.x2.shape[0]):
                    k = kernel(self.x1[i], self.x2[j])
                    assert jnp.isclose(k, K[i,j]), f'{k}, {K[i, j]}'

    def test_gram_matrix_grads(self) -> None:

        for add_linear_kernel in [True, False]:
            length_scale = 2.
            score_estimator = AbstractScoreEstimator(add_linear_kernel=add_linear_kernel)

            def kernel(x1, x2):
                k = self.rbf_kernel(x1, x2, length_scale)
                if add_linear_kernel:
                    k = k + jnp.dot(x1, x2)
                return k

            kernel_grad = jax.grad(kernel, argnums=(0, 1))

            K, grad1, grad2 = score_estimator.grad_gram(self.x1, self.x2, length_scale)

            for i in range(self.x1.shape[0]):
                for j in range(self.x2.shape[0]):
                    k = kernel(self.x1[i], self.x2[j])
                    dx1_k, dx2_k = kernel_grad(self.x1[i], self.x2[j])
                    assert jnp.isclose(k, K[i, j]), f'{k}, {K[i, j]}'
                    assert jnp.all(jnp.isclose(dx1_k, grad1[i, j])), f'{dx1_k}, {grad1[i, j]}'
                    assert jnp.all(jnp.isclose(dx2_k, grad2[i, j])), f'{dx2_k}, {grad2[i, j]}'


class TestSSGE(unittest.TestCase):

    def setUp(self) -> None:
        key = jax.random.PRNGKey(9234)
        key1, key2 = jax.random.split(key)
        self.loc = jnp.array([0.0, 0.0])
        self.scale_diag = jnp.array([1.0, 1.0])
        self.dist = tfp.distributions.MultivariateNormalDiag(loc=self.loc, scale_diag=self.scale_diag)
        self.x_samples = self.dist.sample(100, seed=key1)
        self.x1, self.x2 = jnp.meshgrid(jnp.linspace(-3, 3, 10), jnp.linspace(-3, 3, 10))
        self.x_query = jnp.stack([self.x1.flatten(), self.x2.flatten()], axis=-1)
        self.logprob = lambda x: (dist.log_prob(x).sum(), dist.log_prob(x))

    def test_score_estimation_x_s(self):
        for add_linear_kernel in [True, False]:
            ssge = SSGE(eta=0.1, add_linear_kernel=add_linear_kernel, n_eigen_threshold=0.98)
            score_estimate = ssge.estimate_gradients_s_x(self.x_query, self.x_samples)
            score, logp = jax.grad(self.logprob, has_aux=True)(self.x_query)
            cos_dist = np.mean([spatial.distance.cosine(s1, s2) for s1, s2 in zip(score, score_estimate)])
            assert cos_dist < 0.05, f'cos-dist = {cos_dist}'

    def test_score_estimation_s(self):
        ssge = SSGE(eta=0.1, add_linear_kernel=False, n_eigen_values=50)
        score_estimate = ssge.estimate_gradients_s(self.x_samples)
        score, logp = jax.grad(self.logprob, has_aux=True)(self.x_samples)
        cos_dist = np.mean([spatial.distance.cosine(s1, s2) for s1, s2 in zip(score, score_estimate)])
        assert cos_dist < 0.05, f'cos-dist = {cos_dist}'

        score_estimate = ssge.estimate_gradients_s(self.x_samples[:-3])
        score, logp = jax.grad(self.logprob, has_aux=True)(self.x_samples[:-3])
        cos_dist = np.mean([spatial.distance.cosine(s1, s2) for s1, s2 in zip(score, score_estimate)])
        assert cos_dist < 0.05, f'cos-dist = {cos_dist}'


if __name__ == '__main__':
    unittest.main()