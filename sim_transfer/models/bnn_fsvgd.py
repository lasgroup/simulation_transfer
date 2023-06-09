from collections import OrderedDict
from functools import partial
from typing import List, Optional, Callable, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates import jax as tfp

from sim_transfer.models.bnn import AbstractFSVGD_BNN
from sim_transfer.sims import Domain, HypercubeDomain


class BNN_FSVGD(AbstractFSVGD_BNN):
    """ BNN with FSVGD inference and a GP prior in the function space """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 domain: Domain,
                 rng_key: jax.random.PRNGKey,
                 likelihood_std: float = 0.2,
                 learn_likelihood_std: bool = False,
                 num_particles: int = 10,
                 bandwidth_svgd: float = 0.2,
                 bandwidth_gp_prior: float = 0.2,
                 data_batch_size: int = 16,
                 num_measurement_points: int = 16,
                 num_train_steps: int = 10000,
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
                         lr=lr, weight_decay=weight_decay, domain=domain, bandwidth_svgd=bandwidth_svgd,
                         likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std,
                         normalize_likelihood_std=normalize_likelihood_std)
        self.bandwidth_gp_prior = bandwidth_gp_prior
        self.num_measurement_points = num_measurement_points

        # initialize kernel
        self.kernel_gp_prior = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_gp_prior)

    def _neg_log_posterior(self, pred_raw: jnp.ndarray, likelihood_std: jnp.array, x_stacked: jnp.ndarray,
                           y_batch: jnp.ndarray, train_data_till_idx: int,
                           num_train_points: Union[float, int], key: jax.random.PRNGKey):
        nll = - num_train_points * self._ll(pred_raw, likelihood_std, y_batch, train_data_till_idx)
        neg_log_prior = - self._gp_prior_log_prob(x_stacked, pred_raw, eps=1e-3)
        neg_log_post =  nll + neg_log_prior
        stats = OrderedDict(train_nll_loss=nll, neg_log_prior=neg_log_prior)
        if self.learn_likelihood_std:
            stats['likelihood_std'] = jnp.mean(likelihood_std)
        return neg_log_post, stats

    def _gp_prior_log_prob(self, x: jnp.array, y: jnp.array, eps: float = 1e-3) -> jnp.ndarray:
        k = self.kernel_gp_prior.matrix(x, x) + eps * jnp.eye(x.shape[0])
        dist = tfd.MultivariateNormalFullCovariance(jnp.zeros(x.shape[0]), k)
        return jnp.mean(jnp.sum(dist.log_prob(jnp.swapaxes(y, -1, -2)), axis=-1)) / x.shape[0]


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

    num_train_points = 10

    x_train = jax.random.uniform(key=next(key_iter), shape=(num_train_points,),
                                 minval=domain.l, maxval=domain.u).reshape(-1, 1)
    y_train = fun(x_train)

    x_test = jnp.linspace(domain.l, domain.u, 100).reshape(-1, 1)
    y_test = fun(x_test)

    bnn = BNN_FSVGD(NUM_DIM_X, NUM_DIM_Y, domain=domain, rng_key=next(key_iter), num_train_steps=20000,
                    data_batch_size=10, num_measurement_points=16, normalize_data=True, bandwidth_svgd=1.0,
                    likelihood_std=0.2, learn_likelihood_std=False,
                    bandwidth_gp_prior=0.2, hidden_layer_sizes=[64, 64, 64],
                    normalization_stats=sim.normalization_stats,
                    hidden_activation=jax.nn.tanh)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
        if NUM_DIM_X == 1:
            bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 2000}',
                        domain_l=domain.l[0], domain_u=domain.u[0])
