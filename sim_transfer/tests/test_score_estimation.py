from tensorflow_probability.substrates import jax as tfp
from scipy import spatial
from typing import Union
import jax
import jax.numpy as jnp
import unittest
import numpy as np


from sim_transfer.score_estimation.ssge import SSGE
from sim_transfer.score_estimation.nu_method import NuMethod
from sim_transfer.score_estimation.kde import KDE
from sim_transfer.score_estimation.abstract import GramMatrixMixin
from sim_transfer.modules.metrics import avg_cosine_distance


class TestGramMatrixMixin(unittest.TestCase):

    def setUp(self) -> None:
        key = jax.random.PRNGKey(68)
        key1, key2 = jax.random.split(key)
        self.x1 = jax.random.normal(key1, shape=(3, 2))
        self.x2 = jax.random.normal(key2, shape=(2, 2))

    @staticmethod
    def rbf_kernel(x1: jnp.array, x2: jnp.array, length_scale: Union[float, jnp.array]) -> jnp.array:
        return jnp.exp(- jnp.linalg.norm((x1 - x2) / length_scale) ** 2 / 2)

    @staticmethod
    def imq_kernel(x1: jnp.array, x2: jnp.array, length_scale: Union[float, jnp.array]) -> jnp.array:
        return jax.lax.rsqrt(1 + jnp.linalg.norm((x1 - x2) / length_scale) ** 2)

    def test_gram_matrix_se(self) -> None:
        for add_linear_kernel in [True, False]:
            length_scale = 0.4
            score_estimator = GramMatrixMixin(kernel_type='se', add_linear_kernel=add_linear_kernel)

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

    def test_gram_matrix_imq(self) -> None:
        for add_linear_kernel in [True, False]:
            length_scale = 0.4
            score_estimator = GramMatrixMixin(kernel_type='imq', add_linear_kernel=add_linear_kernel)

            def kernel(x1, x2):
                k = self.imq_kernel(x1, x2, length_scale)
                if add_linear_kernel:
                    k += jnp.dot(x1, x2)
                return k

            K = score_estimator.gram(self.x1, self.x2, length_scale)

            for i in range(self.x1.shape[0]):
                for j in range(self.x2.shape[0]):
                    k = kernel(self.x1[i], self.x2[j])
                    assert jnp.isclose(k, K[i,j]), f'{k}, {K[i, j]}'

    def test_gram_matrix_grads(self) -> None:

        for kernel_type in ['se', 'imq']:
            for add_linear_kernel in [False, True]:
                length_scale = 2.
                score_estimator = GramMatrixMixin(kernel_type=kernel_type, add_linear_kernel=add_linear_kernel)

                def kernel(x1, x2):
                    if kernel_type == 'se':
                        k = self.rbf_kernel(x1, x2, length_scale)
                    elif kernel_type == 'imq':
                        k = self.imq_kernel(x1, x2, length_scale)
                    else:
                        raise NotImplementedError
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
        self.logprob = lambda x: (self.dist.log_prob(x).sum(), self.dist.log_prob(x))

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


class TestNuMethod(unittest.TestCase):

    def setUp(self) -> None:
        key = jax.random.PRNGKey(56756)
        key1, key2 = jax.random.split(key)
        self.loc = jnp.array([10., -5.])
        self.scale_diag = jnp.array([0.5, 2.0])
        self.dist = tfp.distributions.MultivariateNormalDiag(loc=self.loc, scale_diag=self.scale_diag)
        self.x_samples = self.dist.sample(100, seed=key1)
        self.x_query = self.dist.sample(400, seed=key2)
        self.logprob = lambda x: (self.dist.log_prob(x).sum(), self.dist.log_prob(x))

    def test_score_estimation_x_s(self):
        nu_method = NuMethod(lam=1e-4, bandwidth=10.)
        score_estimate = nu_method.estimate_gradients_s_x(queries=self.x_query, samples=self.x_samples)
        score, logp = jax.grad(self.logprob, has_aux=True)(self.x_query)
        cos_dist = np.mean([spatial.distance.cosine(s1, s2) for s1, s2 in zip(score, score_estimate)])
        assert cos_dist < 0.05, f'cos-dist = {cos_dist}'

    def test_score_estimation_x(self):
        nu_method = NuMethod(lam=1e-4, bandwidth=10.)
        score_estimate = nu_method.estimate_gradients_s(x=self.x_samples)
        score, logp = jax.grad(self.logprob, has_aux=True)(self.x_samples)
        cos_dist = np.mean([spatial.distance.cosine(s1, s2) for s1, s2 in zip(score, score_estimate)])
        assert cos_dist < 0.05, f'cos-dist = {cos_dist}'


class TestKDE(unittest.TestCase):

    def setUp(self) -> None:
        key = jax.random.PRNGKey(9234)
        key1, key2 = jax.random.split(key)
        self.loc = jnp.array([0.0, 0.0])
        self.scale_diag = jnp.array([2.0, 2.0])
        self.dist = tfp.distributions.MultivariateNormalDiag(loc=self.loc, scale_diag=self.scale_diag)
        self.x_samples = self.dist.sample(1000, seed=key1)
        self.x1, self.x2 = jnp.meshgrid(jnp.linspace(-4, 4, 10), jnp.linspace(-4, 4, 10))
        self.x_query = jnp.stack([self.x1.flatten(), self.x2.flatten()], axis=-1)
        self.logprob = lambda x: (self.dist.log_prob(x).sum(), self.dist.log_prob(x))

    def test_score_estimation_x_s(self):
        kde = KDE()
        score_estimate = kde.estimate_gradients_s_x(self.x_query, self.x_samples)
        score, logp = jax.grad(self.logprob, has_aux=True)(self.x_query)
        cos_dist = avg_cosine_distance(score, score_estimate)
        assert cos_dist < 0.1, f'cos-dist = {cos_dist}'

    def test_kde_integates_to_1(self):
        dist = tfp.distributions.MultivariateNormalDiag(loc=[-2.], scale_diag=[0.5])
        samples = dist.sample(seed=jax.random.PRNGKey(24234), sample_shape=10)
        kde = KDE()
        query = jnp.linspace(-7, 5, num=200)[:, None]
        ps = jnp.exp(kde.density_estimates_log_prob(query, samples))
        integral = jnp.trapz(x=query.squeeze(-1), y=ps)
        assert (abs(integral) - 1) < 0.01


class TestScoreNetwork(unittest.TestCase):

    def testScoreNetworkSaveLoad(self):
        from sim_transfer.sims.simulators import GaussianProcessSim
        from sim_transfer.score_estimation.score_network_attn import ScoreMatchingEstimator

        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(9234), 3)
        sim = GaussianProcessSim(input_size=2, output_size=2)
        domain = sim.domain

        def sample_ms_f_fn(rng_key: jax.random.PRNGKey, mset_size: int, num_f_samples: int):
            rng_key_mset, rng_key_f = jax.random.split(rng_key, 2)
            mset = domain.sample_uniformly(rng_key_mset, mset_size)
            f_samples = sim.sample_function_vals(x=mset, rng_key=rng_key_f, num_samples=num_f_samples)
            return mset, f_samples

        est = ScoreMatchingEstimator(input_size=2,
                                     output_size=2,
                                     sample_ms_f_fn=sample_ms_f_fn,
                                     mset_size=6,
                                     num_msets_per_step=2,
                                     num_f_samples=100,
                                     rng_key=key1,
                                     loss_mode='mm',
                                     activation_fn=jax.nn.gelu,
                                     weight_decay=0.00)

        xm = domain.sample_uniformly(key2, 6)
        f = sim.sample_function_vals(x=xm, rng_key=key3, num_samples=20)

        est.train(num_iter=5)
        pred1 = est.pred_score(xm, f)

        path = '/tmp/sdkjfsksdjf.pkl'
        est.save_state(path=path)

        est.train(num_iter=5)
        pred2 = est.pred_score(xm, f)
        assert not jnp.allclose(pred1, pred2)

        est.load_state(path=path)
        pred3 = est.pred_score(xm, f)
        assert jnp.allclose(pred1, pred3)


if __name__ == '__main__':
    unittest.main()
