from collections import OrderedDict
from typing import List, Optional, Callable, Dict, Union
from jaxtyping import PyTree

import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

from sim_transfer.models.bnn import AbstractVariationalBNN


class BNN_VI(AbstractVariationalBNN):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 rng_key: jax.random.PRNGKey,
                 likelihood_std: Union[float, jnp.array] = 0.2,
                 learn_likelihood_std: bool = False,
                 num_post_samples: int = 10,
                 data_batch_size: int = 16,
                 num_train_steps: int = 10000,
                 lr: float = 1e-3,
                 kl_prefactor: float = 1.0,
                 normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None,
                 use_prior: bool = True,
                 weight_prior_std: float = 0.5,
                 bias_prior_std: float = 1e1):
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key,
                         data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                         num_batched_nns=num_post_samples, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=normalization_stats,
                         likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std)
        self.use_prior = use_prior
        self.kl_prefactor = kl_prefactor


        # construct the neural network prior distribution
        if use_prior:
            self._prior_dist = self._construct_nn_param_prior(weight_prior_std, bias_prior_std)

        # init learnable variational posterior
        # heuristic for setting the initial of the prior on the weights
        weight_std = 4.0 / jnp.sqrt(jnp.sum(jnp.array(hidden_layer_sizes)))
        _prior = self.batched_model.params_prior(weight_prior_std=weight_std, bias_prior_std=0.5)
        self.params.update({'posterior_mean': _prior.mean(), 'posterior_std': _prior.stddev()})

        # init optimizer for distillation prior
        self.optim = optax.adam(learning_rate=lr)
        self.opt_state = self.optim.init(self.params)

    def _loss(self, params: Dict, x_batch: jnp.array, y_batch: jnp.array,
                    num_train_points: int, key: jax.random.PRNGKey) -> [jnp.ndarray, Dict]:
        # sample NNs from posterior
        posterior_dist = tfd.MultivariateNormalDiag(params['posterior_mean'], params['posterior_std'])
        params_sample = posterior_dist.sample(seed=key, sample_shape=self.num_post_samples)

        # compute data log-likelihood
        likelihood_std = self._likelihood_std_transform(params['likelihood_std_raw']) if self.learn_likelihood_std \
            else self.likelihood_std
        pred_raw = self.batched_model.forward_vec(x_batch, self.batched_model.unravel_batch(params_sample))
        ll = jnp.mean(tfd.MultivariateNormalDiag(pred_raw, likelihood_std).log_prob(y_batch))

        if self.use_prior:
            # estimate KL divergence between posterior and prior
            log_posterior = jnp.mean(posterior_dist.log_prob(params_sample))
            log_prior = jnp.mean(self._prior_dist.log_prob(params_sample))
            kl = log_posterior - log_prior
            loss = -ll + self.kl_prefactor * kl / num_train_points
            stats = OrderedDict(loss=loss, train_nll_loss=-ll, log_posterior=log_posterior,
                                log_prior=log_prior, kl=kl)
        else:
            loss = - ll
            stats = OrderedDict(loss=loss, train_nll_loss=-ll)
        if self.learn_likelihood_std:
            stats['likelihood_std'] = jnp.mean(likelihood_std)
        return loss, stats

    @property
    def posterior_dist(self) -> tfd.Distribution:
        return tfd.MultivariateNormalDiag(self.params['posterior_mean'], self.params['posterior_std'])

    @property
    def prior_dist(self):
        return self._prior_dist


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

    x_train = jax.random.uniform(key=next(key_iter), shape=(num_train_points,),
                                 minval=domain.l, maxval=domain.u).reshape(-1, 1)
    y_train = fun(x_train)

    x_test = jnp.linspace(domain.l, domain.u, 100).reshape(-1, 1)
    y_test = fun(x_test)

    bnn = BNN_VI(NUM_DIM_X, NUM_DIM_Y, next(key_iter), num_train_steps=20000, data_batch_size=5, use_prior=True,
                 kl_prefactor=1e-1, likelihood_std=0.05, hidden_layer_sizes=[64, 64, 64])
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
        bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'BNN-VI, iter {(i + 1) * 5000}',
                    domain_l=domain.l[0], domain_u=domain.u[0])
