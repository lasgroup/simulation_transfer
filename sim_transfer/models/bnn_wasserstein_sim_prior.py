from collections import OrderedDict
from functools import partial
from typing import List, Optional, Callable, Dict, Union, Tuple

import jax
import jax.numpy as jnp

from sim_transfer.models.bnn import AbstractParticleBNN, MeasurementSetMixin
from sim_transfer.modules.metrics import wasserstein_distance
from sim_transfer.sims import FunctionSimulator, Domain


class BNN_Wasserstein_SimPrior(AbstractParticleBNN, MeasurementSetMixin):
    """ BNN based on MMD w.r.t. sim prior. """

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
                 likelihood_std: Union[float, jnp.array] = 0.1,
                 learn_likelihood_std: bool = False,
                 data_batch_size: int = 8,
                 num_train_steps: int = 10000,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-3,
                 normalize_data: bool = True,
                 normalize_likelihood_std: bool = False,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        AbstractParticleBNN.__init__(self, input_size=input_size, output_size=output_size, rng_key=rng_key,
                                     data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                                     num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                                     hidden_activation=hidden_activation, last_activation=last_activation,
                                     normalize_data=normalize_data, normalization_stats=normalization_stats,
                                     normalize_likelihood_std=normalize_likelihood_std, lr=lr, weight_decay=weight_decay,
                                     likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std)
        MeasurementSetMixin.__init__(self, domain=domain)
        self.num_measurement_points = num_measurement_points
        self.independent_output_dims = independent_output_dims

        # check and set function sim
        self.function_sim = function_sim
        assert function_sim.output_size == self.output_size and function_sim.input_size == self.input_size
        self.num_f_samples = num_f_samples

    @partial(jax.jit, static_argnums=(0,))
    def _surrogate_loss(self, params: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                        num_train_points: int, key: jax.random.PRNGKey,
                        f_sim_noise_std: float = 0.05) -> [jnp.ndarray, Dict]:
        key1, key2, key3 = jax.random.split(key, num=3)

        # combine the training data batch with a batch of sampled measurement points
        train_batch_size = x_batch.shape[0]
        x_measurement = self._sample_measurement_points(key1, num_points=self.num_measurement_points)
        x_stacked = jnp.concatenate([x_batch, x_measurement], axis=0)

        # posterior samples
        f_nn = self.batched_model.forward_vec(x_stacked, params['nn_params_stacked'])

        # get likelihood std
        likelihood_std = self._likelihood_std_transform(params['likelihood_std_raw']) if self.learn_likelihood_std \
            else self.likelihood_std

        # negative log-likelihood
        nll = - num_train_points * self._ll(f_nn, likelihood_std, y_batch, train_batch_size)

        # estimate mmd between posterior and prior
        f_sim_samples = self._fsim_samples(x_stacked, key=key2)
        if self.independent_output_dims:
            dist_distance = jnp.sum(
                jax.vmap(wasserstein_distance, in_axes=(-1, -1), out_axes=-1)(f_nn, f_sim_samples))
        else:
            raise NotImplementedError('We support only independent output dimensions for now.')

        loss = nll + dist_distance

        stats = OrderedDict(train_nll_loss=nll, dist_distance=dist_distance, loss=loss)
        if self.learn_likelihood_std:
            stats['likelihood_std'] = jnp.mean(likelihood_std)
        return loss, stats

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
    SIM_TYPE = 'LinearSim'

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

    bnn = BNN_Wasserstein_SimPrior(input_size=NUM_DIM_X, output_size=NUM_DIM_Y, domain=domain, rng_key=next(key_iter),
                                   function_sim=sim, num_particles=20, normalize_data=True,
                                   normalization_stats=sim.normalization_stats, num_measurement_points=16,
                                   weight_decay=0.0, num_f_samples=256, independent_output_dims=True,
                                   likelihood_std=0.05)

    bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=1)
    bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter 1', domain_l=domain.l[0], domain_u=domain.u[0])
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
        bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {i * 5000}',
                    domain_l=domain.l[0], domain_u=domain.u[0])
