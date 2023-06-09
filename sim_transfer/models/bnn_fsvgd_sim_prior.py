from collections import OrderedDict
from typing import List, Optional, Callable, Dict, Union

import jax
import jax.numpy as jnp

from sim_transfer.models.bnn import AbstractFSVGD_BNN
from sim_transfer.score_estimation import SSGE, KDE, NuMethod
from sim_transfer.sims import FunctionSimulator, Domain
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd


class BNN_FSVGD_SimPrior(AbstractFSVGD_BNN):
    _score_estimator_types = ['SSGE', 'ssge', 'nu_method', 'nu-method', 'GP', 'gp', 'KDE', 'kde',
                              'gp+nu_method']

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
                 ssge_kernel_type: str = 'IMQ',
                 bandwidth_score_estim: Optional[float] = None,  # if None, a bandiwidth heuristic is used
                 bandwidth_svgd: float = 0.2,
                 data_batch_size: int = 8,
                 num_train_steps: int = 10000,
                 switch_score_estimator_frac: float = 0.75,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-3,
                 normalize_data: bool = True,
                 normalize_likelihood_std: bool = False,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key,
                         data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                         num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=normalization_stats,
                         normalize_likelihood_std=normalize_likelihood_std,
                         lr=lr, weight_decay=weight_decay,
                         likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std,
                         domain=domain, bandwidth_svgd=bandwidth_svgd)
        self.num_measurement_points = num_measurement_points

        # check and set function sim
        self.function_sim = function_sim
        assert function_sim.output_size == self.output_size and function_sim.input_size == self.input_size
        self.num_f_samples = num_f_samples

        assert score_estimator in self._score_estimator_types, \
            f'score_estimator must be one of {self._score_estimator_types}'
        if len(score_estimator.split('+')) == 2:
            # switch score estimator after some time
            self.score_estimator = score_estimator.split('+')[0]
            self.score_estimator_switch = score_estimator.split('+')[1]
            self.switch_score_estimator_at_iter = int(switch_score_estimator_frac * num_train_steps)
        else:
            self.score_estimator = score_estimator
            self.score_estimator_switch = None
            self.switch_score_estimator_at_iter = jnp.inf
        self.independent_output_dims = independent_output_dims
        self.bandwidth_score_estim = bandwidth_score_estim
        self.ssge_kernel_type = ssge_kernel_type

        self._itr = 0

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        self.opt_state, self.params, stats = self._step_jit(self.opt_state, self.params, x_batch, y_batch,
                                                            key=self.rng_key, num_train_points=num_train_points)
        self._itr += 1

        # functionality for switching score estimators
        if self.switch_score_estimator_at_iter == self._itr:
            print(f'Switching score estimator from {self.score_estimator} to {self.score_estimator_switch}')
            self.score_estimator = self.score_estimator_switch
            self._before_training_loop_callback()  # re-run the callback to re-initialize the score estimator
            self._step_jit = jax.jit(self._step)  # re-compile the step function
        return stats

    def _neg_log_posterior(self, pred_raw: jnp.ndarray, likelihood_std: jnp.array, x_stacked: jnp.ndarray,
                            y_batch: jnp.ndarray, train_data_till_idx: int,
                            num_train_points: Union[float, int], key: jax.random.PRNGKey):
        nll = - num_train_points * self._ll(pred_raw, likelihood_std, y_batch, train_data_till_idx)
        if self.score_estimator in ['SSGE', 'ssge', 'nu_method', 'nu-method']:
            prior_score = self._estimate_prior_score(pred_raw, x_stacked, key)
            neg_log_post = nll - jnp.sum(jnp.mean(pred_raw * jax.lax.stop_gradient(prior_score), axis=-2))
        elif self.score_estimator in ['GP', 'gp']:
            prior_logprob = self._prior_log_prob_gp_approx(pred_raw, x_stacked, key)
            neg_log_post = nll - prior_logprob
        elif self.score_estimator in ['KDE', 'kde']:
            prior_logprob = self._prior_log_prob_kde_approx(pred_raw, x_stacked, key)
            neg_log_post = nll - prior_logprob
        else:
            raise NotImplementedError
        stats = OrderedDict(train_nll_loss=nll)
        if self.learn_likelihood_std:
            stats['likelihood_std'] = jnp.mean(likelihood_std)
        return neg_log_post, stats

    def _estimate_prior_score(self, pred_raw: jnp.ndarray, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Uses a non-parametric score estimator to estimate the prior marginals' score of the function simulator.
        """
        f_samples = self._fsim_samples(x, key)
        if self.independent_output_dims:
            # performs score estimation for each output dimension independently
            score_estimate = self._estimate_gradients_s_x_vectorized(pred_raw, f_samples)
        else:
            # performs score estimation for all output dimensions jointly
            # before score estimation call, flatten the output dimensions
            score_estimate = self._score_estim.estimate_gradients_s_x(pred_raw.reshape((pred_raw.shape[0], -1)),
                                                                     f_samples.reshape((f_samples.shape[0], -1)))
            # add back the output dimensions
            score_estimate = score_estimate.reshape(pred_raw.shape)
        assert score_estimate.shape == pred_raw.shape
        return score_estimate

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
        if self.independent_output_dims:
            prior_logprob = self._estimate_log_probs_vectorized(pred_raw, f_samples)
        else:
            prior_logprob = self.kde.density_estimates_log_prob(pred_raw.reshape((pred_raw.shape[0], -1)),
                                                                f_samples.reshape((f_samples.shape[0], -1)))
        return jnp.sum(prior_logprob)

    def _fsim_samples(self, x: jnp.array, key: jax.random.PRNGKey) -> jnp.ndarray:
        x_unnormalized = self._unnormalize_data(x)
        f_prior = self.function_sim.sample_function_vals(x=x_unnormalized, num_samples=self.num_f_samples, rng_key=key)
        f_prior_normalized = self._normalize_y(f_prior)
        return f_prior_normalized

    def _before_training_loop_callback(self) -> None:
        """
        Called right before the training loop starts. Is used to
        1) estimate the bandwidth of the score estimator if necessary
        2) initialize the score estimator
        3) set up vectorized score estimation if necessary
        """

        # in case of SSGE or Nu-method, estimate bandwidth with mean heuristic if not provided
        if self.score_estimator in ['ssge', 'SSGE', 'nu_method', 'nu-method'] and self.bandwidth_score_estim is None:
            self.bandwidth_score_estim = self._bandwidth_heuristic()
            if self.score_estimator in ['nu_method', 'nu-method']:
                # heuristic for nu-method since curl-free kernels require larger bandwidth
                self.bandwidth_score_estim *= 4
            print('Estimated bandwidth via median heuristic: ', self.bandwidth_score_estim)

        # initialize score estimator
        if self.score_estimator in ['SSGE', 'ssge']:
            self._score_estim = SSGE(bandwidth=self.bandwidth_score_estim, kernel_type=self.ssge_kernel_type)
        elif self.score_estimator in ['nu_method', 'nu-method']:
            self._score_estim = NuMethod(lam=1e-4, bandwidth=self.bandwidth_score_estim)
        elif self.score_estimator in ['GP', 'gp']:
            pass
        elif self.score_estimator in ['KDE', 'kde']:
            self.kde = KDE(bandwidth=self.bandwidth_score_estim)
        else:
            raise ValueError(f'Unknown score_estimator {self.score_estimator}. Must be either SSGE or GP')

        # in case of independent output dimensions, vectorize the score estimation function over the output dimension
        if self.independent_output_dims:
            # set up vectorized version of the score estimation function
            if self.score_estimator in ['SSGE', 'ssge', 'nu_method', 'nu-method']:
                # create estimate gradient function over the output dims
                self._estimate_gradients_s_x_vectorized = jax.vmap(
                    lambda y, f: self._score_estim.estimate_gradients_s_x(y, f), in_axes=-1, out_axes=-1)
            elif self.score_estimator in ['KDE', 'kde']:
                    self._estimate_log_probs_vectorized = jax.vmap(
                        lambda y, f: self.kde.density_estimates_log_prob(y, f), in_axes=-1, out_axes=-1)

    def _bandwidth_heuristic(self) -> Union[float, jnp.array]:
        """
        Estimates the median distance between f_samples from the sim prior to use as bandwidth
        for SSGE or the nu-method.
        """
        bandwidth_median_list = []
        for i in range(50):
            x = self._sample_measurement_points(self.rng_key, num_points=self.num_measurement_points)
            f_samples = self._fsim_samples(x, key=self.rng_key)
            if self.independent_output_dims:
                bandwidth_median = jnp.mean(
                    jax.vmap(SSGE.bandwith_median_heuristic, in_axes=-1, out_axes=-1)(f_samples))
            else:
                bandwidth_median = SSGE.bandwith_median_heuristic(f_samples.reshape(f_samples.shape[0], -1))
            bandwidth_median_list.append(bandwidth_median)
        return jnp.mean(jnp.stack(bandwidth_median_list))


if __name__ == '__main__':
    from sim_transfer.sims import SinusoidsSim, QuadraticSim, LinearSim

    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key

    key_iter = key_iter()
    NUM_DIM_X = 1
    NUM_DIM_Y = 2
    SIM_TYPE = 'SinusoidsSim'

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
    score_estimator = 'gp'

    x_train = jax.random.uniform(key=next(key_iter), shape=(num_train_points,),
                                 minval=domain.l, maxval=domain.u).reshape(-1, 1)
    y_train = fun(x_train)

    x_test = jnp.linspace(domain.l, domain.u, 100).reshape(-1, 1)
    y_test = fun(x_test)

    bnn = BNN_FSVGD_SimPrior(NUM_DIM_X, NUM_DIM_Y, domain=domain, rng_key=next(key_iter), function_sim=sim,
                             hidden_layer_sizes=[64, 64, 64], num_train_steps=4000, data_batch_size=4,
                             num_particles=20, num_f_samples=256, num_measurement_points=16,
                             bandwidth_svgd=1., bandwidth_score_estim=1.0, ssge_kernel_type='IMQ',
                             normalization_stats=sim.normalization_stats, likelihood_std=0.05,
                             score_estimator=score_estimator)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
        if NUM_DIM_X == 1:
            bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'FSVGD SimPrior {score_estimator}, iter {(i + 1) * 2000}',
                        domain_l=domain.l[0], domain_u=domain.u[0])
