from collections import OrderedDict
from functools import partial
from typing import List, Optional, Callable, Dict, Union, Tuple
from jaxtyping import PyTree

import time
import jax
import jax.numpy as jnp
import optax

from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd

from sim_transfer.sims import FunctionSimulator, Domain, HypercubeDomain
from sim_transfer.models.bnn import AbstractSVGD_BNN, MeasurementSetMixin
from sim_transfer.modules.util import aggregate_stats
from sim_transfer.modules.metrics import mmd2


class BNN_SVGD_DistillPrior(AbstractSVGD_BNN, MeasurementSetMixin):
    """ BNN with SVGD inference whose prior is distilled from a simulator """

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
                 likelihood_std: float = 0.2,
                 learn_likelihood_std: bool = False,
                 bandwith_mmd_range_log2: Tuple[int, int] = (-3, 6),
                 bandwidth_svgd: float = 10.0,
                 data_batch_size: int = 8,
                 num_train_steps: int = 10000,
                 num_distill_steps: int = 10000,
                 lr: float = 1e-3,
                 lr_distill_prior: float = 1e-3,
                 normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        AbstractSVGD_BNN.__init__(self, input_size=input_size, output_size=output_size, rng_key=rng_key,
                                  data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                                  num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                                  hidden_activation=hidden_activation, last_activation=last_activation,
                                  normalize_data=normalize_data, normalization_stats=normalization_stats,
                                  lr=lr, likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std,
                                  bandwidth_svgd=bandwidth_svgd, use_prior=True)
        MeasurementSetMixin.__init__(self, domain=domain)

        self.num_measurement_points = num_measurement_points
        self.independent_output_dims = independent_output_dims
        self.num_distill_steps = num_distill_steps

        # init learnable distillation prior
        # heuristic for setting the std of the prior on the weights
        weight_std = 4.0 / jnp.sqrt(jnp.sum(jnp.array(hidden_layer_sizes)))
        _prior = self.batched_model.params_prior(weight_prior_std=weight_std, bias_prior_std=0.5)
        self.distill_prior_params = {'mean': _prior.mean(), 'std': _prior.stddev()}

        # init optimizer for distillation prior
        self.optim_prior = optax.adam(learning_rate=lr_distill_prior)
        self.opt_state_prior = self.optim.init(self.distill_prior_params)

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

    def fit(self, *args, num_distill_steps: Optional[int] = None, log_period: int = 1000, **kwargs):
        if num_distill_steps is not None:
            num_distill_steps = self.num_distill_steps
        print('\n----- Distilling prior...')
        self.fit_distill_prior(num_steps=num_distill_steps, log_period=log_period)
        print('\n----- Approximating posterior via SVGD...')
        self.fit_posterior(*args, log_period=log_period, **kwargs)

    def fit_posterior(self, *args, **kwargs):
        super().fit(*args, **kwargs)

    def fit_distill_prior(self, num_steps: int = 1000, log_period: int = 1000):
        stats_list = []
        t_start_period = time.time()
        steps_cum_period = 0

        for step in range(0, num_steps):
            stats = self.step_distill()
            stats_list.append(stats)
            steps_cum_period += 1

            if step % log_period == 0 or step == 1:
                duration_sec = time.time() - t_start_period
                duration_per_step_ms = duration_sec / steps_cum_period * 1000
                stats_agg = aggregate_stats(stats_list)

                stats_msg = ' | '.join([f'{n}: {v:.4f}' for n, v in stats_agg.items()])
                msg = (f'Distill Step {step}/{num_steps} | {stats_msg} | Duration {duration_sec:.2f} sec | '
                       f'Time per step {duration_per_step_ms:.2f} ms')
                print(msg)

                stats_list = []
                steps_cum_period = 0
                t_start_period = time.time()

    @property
    def prior_dist(self):
        return tfd.MultivariateNormalDiag(loc=self.distill_prior_params['mean'],
                                          scale_diag=self.distill_prior_params['std'])

    def step_distill(self):
        self.opt_state_prior, self.distill_prior_params, stats = self._step_distill_jit(
            self.opt_state_prior, self.distill_prior_params, self.rng_key)
        return stats

    def plot_distill_prior_samples(self, key: Optional[jax.random.PRNGKey] = None,
                                   domain_l: Optional[float] = None,
                                   domain_u: Optional[float] = None,
                                   title: str = 'Distilled prior samples',):
        assert self.input_size == 1 and self.output_size == 1
        from matplotlib import pyplot as plt

        x_plot = jnp.linspace(domain_l, domain_u, 200).reshape((-1, 1))
        x_plot = self._normalize_data(x_plot)

        preds = self._random_distill_prior_fwd(x_plot, sample_shape=(self.num_particles,), key=key)
        preds = self._unnormalize_y(preds)
        fig, ax = plt.subplots()
        for i in range(preds.shape[0]):
            ax.plot(x_plot, preds[i, :, 0])
        fig.suptitle(title)
        fig.show()
        return fig, ax

    def _random_distill_prior_fwd(self, x: jnp.ndarray, sample_shape: Tuple,
                                  distill_prior_params: Optional[PyTree] = None,
                                  key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """ Sample from random prior. """
        if key is None:
            key = self.rng_key
        if distill_prior_params is None:
            distill_prior_params = self.distill_prior_params
        distill_prior_dist = tfd.MultivariateNormalDiag(loc=distill_prior_params['mean'],
                                                        scale_diag=distill_prior_params['std'])
        sampled_params = self.batched_model.unravel_batch(
            distill_prior_dist.sample(sample_shape=sample_shape, seed=key))
        return self.batched_model.forward_vec(x, sampled_params)

    def _fsim_samples(self, x: jnp.array, key: jax.random.PRNGKey) -> jnp.ndarray:
        x_unnormalized = self._unnormalize_data(x)
        f_prior = self.function_sim.sample_function_vals(x=x_unnormalized, num_samples=self.num_f_samples, rng_key=key)
        f_prior_normalized = self._normalize_y(f_prior)
        return f_prior_normalized

    @partial(jax.jit, static_argnums=(0,))
    def _step_distill_jit(self, opt_state: optax.OptState, distill_prior_params: PyTree, key: jax.random.PRNGKey):
        (loss, stats), grad = jax.value_and_grad(self._loss_distill_prior, has_aux=True)(
            distill_prior_params, key)
        updates, opt_state = self.optim.update(grad, opt_state, distill_prior_params)
        distill_prior_params = optax.apply_updates(distill_prior_params, updates)
        return opt_state, distill_prior_params, stats

    def _loss_distill_prior(self, distill_prior_params: PyTree, key: jax.random.PRNGKey,
                            f_sim_noise_std: float = 0.05) -> Tuple[jnp.ndarray, Dict]:
        """ Loss for distillation prior. """
        key1, key2, key3 = jax.random.split(key, num=3)
        x_measurement = self._sample_measurement_points(key1, num_points=self.num_measurement_points)

        f_distill_prior = self._random_distill_prior_fwd(x_measurement, sample_shape=(self.num_particles,),
                                                        distill_prior_params=distill_prior_params, key=key2)
        f_sim_prior = self._fsim_samples(x_measurement, key=key2)
        f_sim_prior += jax.random.normal(key3, shape=f_sim_prior.shape) * f_sim_noise_std
        # add small amount of noise

        if self.independent_output_dims:
            mmd = jnp.sum(self._mmd_fn_vmap(f_distill_prior, f_sim_prior))
        else:
            mmd = jnp.sum(self._mmd_fn(f_distill_prior.reshape(f_distill_prior.shape[0], -1),
                                       f_sim_prior.reshape(f_sim_prior.shape[0], -1)))

        stats = OrderedDict(mmd=mmd)
        return mmd, stats



if __name__ == '__main__':
    from sim_transfer.sims import GaussianProcessSim, SinusoidsSim

    def key_iter():
        key = jax.random.PRNGKey(9836)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key

    key_iter = key_iter()
    NUM_DIM_X = 1
    NUM_DIM_Y = 1
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
        bnn = BNN_SVGD_DistillPrior(NUM_DIM_X, NUM_DIM_Y, domain=domain, rng_key=next(key_iter), function_sim=sim,
                                    hidden_layer_sizes=[128] * 3,
                                    num_particles=30,
                                    data_batch_size=4,
                                    num_f_samples=400,
                                    num_measurement_points=8,
                                    lr_distill_prior=1e-2,
                                    bandwidth_svgd=0.1,
                                    independent_output_dims=False)
        print('\n----- Distilling prior...')
        for i in range(5):
            bnn.fit_distill_prior(num_steps=5000, log_period=500)
            bnn.plot_distill_prior_samples(key=next(key_iter), domain_l=domain.l, domain_u=domain.u,
                                           title=f'Distillation prior samples, step {i * 5000}')
        print('\n----- Approximating posterior via SVGD...')
        for i in range(5):
            bnn.fit_posterior(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
            bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 2000}')
