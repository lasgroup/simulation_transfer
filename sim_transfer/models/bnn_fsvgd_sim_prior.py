from collections import OrderedDict
from typing import List, Optional, Callable, Dict, Union

import jax
import jax.numpy as jnp

from sim_transfer.models.bnn import AbstractFSVGD_BNN
from sim_transfer.score_estimation import SSGE
from sim_transfer.sims import FunctionSimulator, Domain
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.math.psd_kernels.internal.util import pairwise_square_distance_matrix
import tensorflow_probability.substrates.jax.distributions as tfd


class BNN_FSVGD_SimPrior(AbstractFSVGD_BNN):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 domain: Domain,
                 rng_key: jax.random.PRNGKey,
                 function_sim: FunctionSimulator,
                 independent_output_dims: bool = True,
                 num_particles: int = 10,
                 num_f_samples: int = 64,
                 num_measurement_points: int = 8,
                 likelihood_std: Union[float, jnp.array] = 0.2,
                 learn_likelihood_std: bool = False,
                 score_estimator: str = 'SSGE',
                 bandwidth_ssge: float = 1.,
                 bandwidth_kde: Optional[float] = None,  # if None, use Scott's rule of thumb for choosing the bandwidth
                 bandwidth_svgd: float = 0.2,
                 data_batch_size: int = 8,
                 num_train_steps: int = 10000,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-3,
                 normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key,
                         data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                         num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=normalization_stats,
                         lr=lr, weight_decay=weight_decay,
                         likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std,
                         domain=domain, bandwidth_svgd=bandwidth_svgd)
        self.num_measurement_points = num_measurement_points

        # check and set function sim
        self.function_sim = function_sim
        assert function_sim.output_size == self.output_size and function_sim.input_size == self.input_size
        self.num_f_samples = num_f_samples

        self.score_estimator = score_estimator
        self.independent_output_dims = independent_output_dims
        if score_estimator in ['SSGE', 'ssge']:
            # initialize ssge algo
            self.ssge = SSGE(bandwidth=bandwidth_ssge)

            # create estimate gradient function over the output dims
            if self.independent_output_dims:
                self.estimate_gradients_s_x_vectorized = jax.vmap(lambda y, f: self.ssge.estimate_gradients_s_x(y, f),
                                                                  in_axes=-1, out_axes=-1)
        elif score_estimator in ['GP', 'gp']:
            pass
        elif score_estimator in ['KDE', 'kde']:
            self.bandwidth_kde = bandwidth_kde
            if self.independent_output_dims:
                self._squared_dist_vmap = jax.vmap(lambda x, y: pairwise_square_distance_matrix(x, y, 1),
                                                   in_axes=-1, out_axes=-1)
        else:
            raise ValueError(f'Unknown score_estimator {score_estimator}. Must be either SSGE or GP')

    def _neg_log_posterior(self, pred_raw: jnp.ndarray, likelihood_std: jnp.array, x_stacked: jnp.ndarray,
                            y_batch: jnp.ndarray, train_data_till_idx: int,
                            num_train_points: Union[float, int], key: jax.random.PRNGKey):
        nll = - self._ll(pred_raw, likelihood_std, y_batch, train_data_till_idx)
        if self.score_estimator in ['SSGE', 'ssge']:
            prior_score = self._estimate_prior_score(x_stacked, pred_raw, key) / num_train_points
            neg_log_post = nll - jnp.sum(jnp.mean(pred_raw * jax.lax.stop_gradient(prior_score), axis=-2))
        elif self.score_estimator in ['GP', 'gp']:
            prior_logprob = self._prior_log_prob_gp_approx(pred_raw, x_stacked, key)
            neg_log_post = nll - prior_logprob / num_train_points
        elif self.score_estimator in ['KDE', 'kde']:
            prior_logprob = self._prior_log_prob_kde_approx(pred_raw, x_stacked, key)
            neg_log_post = nll - prior_logprob / num_train_points
        else:
            raise NotImplementedError
        stats = OrderedDict(train_nll_loss=nll)
        if self.learn_likelihood_std:
            stats['likelihood_std'] = jnp.mean(likelihood_std)
        return neg_log_post, stats

    def _estimate_prior_score(self, x: jnp.array, y: jnp.array, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Uses SSGE to estimate the prior marginals' score of the function simulator.
        """
        f_samples = self._fsim_samples(x, key)
        if self.independent_output_dims:
            # performs score estimation for each output dimension independently
            ssge_score = self.estimate_gradients_s_x_vectorized(y, f_samples)
        else:
            # performs score estimation for all output dimensions jointly
            # befor score estimation call, flatten the output dimensions
            ssge_score = self.ssge.estimate_gradients_s_x(y.reshape((y.shape[0], -1)),
                                                          f_samples.reshape((f_samples.shape[0], -1)))
            # add back the output dimensions
            ssge_score = ssge_score.reshape(y.shape)
        assert ssge_score.shape == y.shape
        return ssge_score

    def _prior_log_prob_gp_approx(self, pred_raw: jnp.ndarray, x: jnp.ndarray, key: jax.random.PRNGKey,
                                  eps: float = 1e-4) -> jnp.ndarray:
        """
        Samples from function_sim and approximates the corresponding marginals as multivariate
        gaussian distributions. Then computes the log probability of pred_raw w.r.t. the estimated GP marginal dist.
        """
        f_samples = self._fsim_samples(x, key)
        if self.independent_output_dims:
            f_mean = jnp.mean(f_samples, axis=0).T
            f_cov = jnp.swapaxes(tfp.stats.covariance(f_samples, sample_axis=0, event_axis=1), 0, -1)
            prior_gp_approx = tfd.MultivariateNormalFullCovariance(
                loc=f_mean, covariance_matrix=f_cov + eps * jnp.eye(x.shape[0]))
            prior_logprob = jnp.sum(prior_gp_approx.log_prob(pred_raw.swapaxes(-1, -2)), axis=(-2, -1))
        else:
            f_samples = f_samples.reshape((self.num_f_samples, -1))
            f_mean = jnp.mean(f_samples, axis=0)
            f_cov = tfp.stats.covariance(f_samples, sample_axis=0, event_axis=1)
            prior_gp_approx = tfd.MultivariateNormalFullCovariance(
                loc=f_mean, covariance_matrix=f_cov + eps * jnp.eye(x.shape[0] * self.output_size))
            prior_logprob = jnp.sum(prior_gp_approx.log_prob(pred_raw.reshape(pred_raw.shape[0], -1)), axis=0)
        return prior_logprob

    def _prior_log_prob_kde_approx(self, pred_raw: jnp.ndarray, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Uses kernel density estimation (KDE) with Gaussian / RBF kernels to approximate the prior score
        """
        f_samples = self._fsim_samples(x, key)
        n, d = f_samples.shape[0], f_samples.shape[1]
        if self.bandwidth_kde == None:
            bandwidth = n ** (-1. / (d + 4))  # Scott's rule of thumb
        else:
            bandwidth = self.bandwidth_kde
        if self.independent_output_dims:
            dists = self._squared_dist_vmap(pred_raw, f_samples) / (2 * bandwidth ** 2)
        else:
            dists = pairwise_square_distance_matrix(pred_raw.reshape((pred_raw.shape[0], -1)),
                                                    f_samples.reshape((f_samples.shape[0], -1)), 1)
        prior_logprob = jnp.sum(jax.scipy.special.logsumexp(-dists, axis=1) - jnp.log(n * bandwidth)
                                - 0.5 * jnp.log(2 * jnp.pi))
        return prior_logprob

    def _fsim_samples(self, x: jnp.array, key: jax.random.PRNGKey) -> jnp.ndarray:
        x_unnormalized = self._unnormalize_data(x)
        f_prior = self.function_sim.sample_function_vals(x=x_unnormalized, num_samples=self.num_f_samples, rng_key=key)
        f_prior_normalized = self._normalize_y(f_prior)
        return f_prior_normalized


if __name__ == '__main__':
    from sim_transfer.sims import SinusoidsSim, QuadraticSim, LinearSim


    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key

    key_iter = key_iter()
    NUM_DIM_X = 1
    NUM_DIM_Y = 1
    SIM_TYPE = 'QuadraticSim'

    if SIM_TYPE == 'QuadraticSim':
        sim = QuadraticSim()
        fun = lambda x: (x - 2) ** 2
    elif SIM_TYPE == 'LinearSim':
        sim = LinearSim()
        fun = lambda x: x
    elif SIM_TYPE == 'SinusoidsSim':
        sim = SinusoidsSim(output_size=NUM_DIM_Y)

        if NUM_DIM_X == 1 and NUM_DIM_Y == 1:
            fun = lambda x: (2 * x + 2 * jnp.sin(2 * x)).reshape(-1, 1)
        elif NUM_DIM_X == 1 and NUM_DIM_Y == 2:
            fun = lambda x: jnp.concatenate([(2 * x + 2 * jnp.sin(2 * x)).reshape(-1, 1),
                                             (- 2 * x + 2 * jnp.cos(1.5 * x)).reshape(-1, 1)], axis=-1)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    domain = sim.domain
    x_measurement = jnp.linspace(domain.l[0], domain.u[0], 50).reshape(-1, 1)

    num_train_points = 3

    x_train = jax.random.uniform(key=next(key_iter), shape=(num_train_points,),
                                 minval=domain.l, maxval=domain.u).reshape(-1, 1)
    y_train = fun(x_train)

    x_test = jnp.linspace(domain.l, domain.u, 100).reshape(-1, 1)
    y_test = fun(x_test)

    bnn = BNN_FSVGD_SimPrior(NUM_DIM_X, NUM_DIM_Y, domain=domain, rng_key=next(key_iter), function_sim=sim,
                             hidden_layer_sizes=[64, 64, 64], num_train_steps=20000, data_batch_size=4,
                             learn_likelihood_std=False, num_f_samples=128, bandwidth_svgd=1.0, bandwidth_ssge=1.0,
                             normalization_stats=sim.normalization_stats,
                             score_estimator='gp')
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
        if NUM_DIM_X == 1:
            bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 2000}',
                        domain_l=domain.l[0], domain_u=domain.u[0])
