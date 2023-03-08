from collections import OrderedDict
from functools import partial
from typing import List, Optional, Callable, Dict, Union, Tuple

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from sim_transfer.sims import FunctionSimulator, Domain, HypercubeDomain
from sim_transfer.models.bnn import AbstractParticleBNN, MeasurementSetMixin
from sim_transfer.modules.util import mmd2


class BNN_MMD_SimPrior(AbstractParticleBNN, MeasurementSetMixin):
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
                 likelihood_std: Union[float, jnp.array] = 0.2,
                 bandwith_mmd_range_log2: Tuple[int, int] = (-3, 6),
                 data_batch_size: int = 8,
                 num_train_steps: int = 10000,
                 lr: float = 1e-3,
                 normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None,
                 log_wandb: bool = False):
        AbstractParticleBNN.__init__(self, input_size=input_size, output_size=output_size, rng_key=rng_key,
                                     data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                                     num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                                     hidden_activation=hidden_activation, last_activation=last_activation,
                                     normalize_data=normalize_data, normalization_stats=normalization_stats,
                                     log_wandb=log_wandb, lr=lr, likelihood_std=likelihood_std)
        MeasurementSetMixin.__init__(self, domain=domain)
        self.num_measurement_points = num_measurement_points
        self.independent_output_dims = independent_output_dims

        # check and set function sim
        self.function_sim = function_sim
        assert function_sim.output_size == self.output_size and function_sim.input_size == self.input_size
        self.num_f_samples = num_f_samples

        # initialize MMD kernel
        self.kernel_mmd = tfp.math.psd_kernels.ExponentiatedQuadratic(
            length_scale=2. ** jnp.arange(*bandwith_mmd_range_log2))

        # setup vectorized MMD functon
        self._mmd_fn = partial(mmd2, kernel=self.kernel_mmd)
        if self.independent_output_dims:
            self._mmd_fn_vmap = jax.vmap(self._mmd_fn, in_axes=(-1, -1), out_axes=-1)

    @partial(jax.jit, static_argnums=(0,))
    def _surrogate_loss(self, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                        num_train_points: int, key: jax.random.PRNGKey,
                        f_sim_noise_std: float = 0.05) -> [jnp.ndarray, Dict]:
        key1, key2, key3 = jax.random.split(key, num=3)
        # combine the training data batch with a batch of sampled measurement points
        train_batch_size = x_batch.shape[0]
        x_domain = self._sample_measurement_points(key1, num_points=self.num_measurement_points)
        x_stacked = jnp.concatenate([x_batch, x_domain], axis=0)

        # posterior samples
        f_raw = self.batched_model.forward_vec(x_stacked, param_vec_stack)

        # negative log-likelihood
        nll = self._nll(f_raw, y_batch, train_batch_size)

        # estimate mmd between posterior and prior
        fsim_samples = self._fsim_samples(x_stacked, key=key2)  # sample function values from sim prior
        f_raw_noisy = f_raw + f_sim_noise_std * jax.random.normal(key2, shape=f_raw.shape)  # add noise to sim prior samples
        if self.independent_output_dims:
            mmd = jnp.sum(self._mmd_fn_vmap(f_raw_noisy, fsim_samples))
        else:
            mmd = jnp.sum(self._mmd_fn(f_raw_noisy.reshape(f_raw_noisy.shape[0], -1),
                                       fsim_samples.reshape(fsim_samples.shape[0], -1)))

        loss = nll + mmd / num_train_points

        stats = OrderedDict(train_nll_loss=nll, mmd=mmd, loss=loss)
        return loss, stats

    def _fsim_samples(self, x: jnp.array, key: jax.random.PRNGKey) -> jnp.ndarray:
        x_unnormalized = self._unnormalize_data(x)
        f_prior = self.function_sim.sample_function_vals(x=x_unnormalized, num_samples=self.num_f_samples, rng_key=key)
        f_prior_normalized = self._normalize_y(f_prior)
        return f_prior_normalized


if __name__ == '__main__':
    from sim_transfer.sims import GaussianProcessSim, SinusoidsSim

    def key_iter():
        key = jax.random.PRNGKey(9836)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key

    key_iter = key_iter()
    NUM_DIM_X = 1
    NUM_DIM_Y = 2
    num_train_points = 2

    if NUM_DIM_X == 1 and NUM_DIM_Y == 1:
        fun = lambda x: (2 * x + 2 * jnp.sin(2 * x)).reshape(-1, 1)
    elif NUM_DIM_X == 1 and NUM_DIM_Y == 2:
        fun = lambda x: jnp.concatenate([(2 * x + 2 * jnp.sin(2 * x)).reshape(-1, 1),
                                         (- 2 * x + 2 * jnp.cos(1.5 * x)).reshape(-1, 1)], axis=-1)
    else:
        raise NotImplementedError

    domain = HypercubeDomain(lower=jnp.array([-7.] * NUM_DIM_X), upper=jnp.array([7.] * NUM_DIM_X))

    x_train = jax.random.uniform(next(key_iter), shape=(num_train_points, NUM_DIM_X), minval=-5, maxval=5)
    y_train = fun(x_train) + 0.01 * jax.random.normal(next(key_iter), shape=(x_train.shape[0], NUM_DIM_Y))

    num_test_points = 100
    x_test = jax.random.uniform(next(key_iter), shape=(num_test_points, NUM_DIM_X), minval=-5, maxval=5)
    y_test = fun(x_test) + 0.1 * jax.random.normal(next(key_iter), shape=(x_test.shape[0], NUM_DIM_Y))

    with jax.disable_jit(False):
        # sim = GaussianProcessSim(input_size=1, output_scale=1.0, mean_fn=lambda x: 2 * x)
        sim = SinusoidsSim(input_size=1, output_size=NUM_DIM_Y)
        bnn = BNN_MMD_SimPrior(NUM_DIM_X, NUM_DIM_Y, domain=domain, rng_key=next(key_iter), function_sim=sim,
                               hidden_layer_sizes=[64, 64, 64],
                               num_particles=30,
                               data_batch_size=4,
                               num_f_samples=400,
                               num_measurement_points=8,
                               independent_output_dims=False)
        for i in range(10):
            bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
            if NUM_DIM_X == 1:
                bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 5000}',
                            domain_l=-7, domain_u=7)
