import jax
import jax.numpy as jnp
import numpy as np
import optax

from typing import List, Optional, Callable, Tuple, Dict, Union
from collections import OrderedDict
from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.sims import FunctionSimulator
from sim_transfer.score_estimation import SSGE

from functools import partial
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd


class BNN_FSVGD_Sim_Prior(BatchedNeuralNetworkModel):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 domain_l: jnp.ndarray,
                 domain_u: jnp.ndarray,
                 rng_key: jax.random.PRNGKey,
                 function_sim: FunctionSimulator,
                 independent_output_dims: bool = True,
                 num_particles: int = 10,
                 num_f_samples: int = 64,
                 num_measurement_points: int = 8,
                 likelihood_std: float = 0.2,
                 bandwidth_ssge: float = 1.,
                 bandwidth_svgd: float = 0.2,
                 data_batch_size: int = 8,
                 num_train_steps: int = 10000,
                 lr=1e-3,
                 normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key,
                         data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                         num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=normalization_stats)
        self.likelihood_std = likelihood_std * jnp.ones(output_size)
        self.num_particles = num_particles
        self.bandwidth_svgd = bandwidth_svgd
        self.num_measurement_points = num_measurement_points

        # check and set function sim
        self.function_sim = function_sim
        assert function_sim.output_size == self.output_size and function_sim.input_size == self.input_size
        self.num_f_samples = num_f_samples

        # check and set domain boundaries
        assert domain_u.shape == domain_l.shape == (self.input_size,)
        assert jnp.all(domain_l <= domain_u), 'lower bound of domain must be smaller than upper bound'
        self.domain_l = domain_l
        self.domain_u = domain_u

        # initialize batched NN
        self.params_stack = self.batched_model.param_vectors_stacked

        # initialize optimizer
        self.optim = optax.adam(learning_rate=lr)
        self.opt_state = self.optim.init(self.params_stack)

        # initialize kernel and ssge algo
        self.kernel_svgd = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_svgd)
        self.ssge = SSGE(bandwidth=bandwidth_ssge)

        # create estimate gradient function over the output dims
        self.independent_output_dims = independent_output_dims
        if independent_output_dims:
            self.estimate_gradients_s_x_vectorized = jax.vmap(lambda y, f: self.ssge.estimate_gradients_s_x(y, f),
                                                              in_axes=-1,
                                                              out_axes=-1)

    def _sample_measurement_points(self, key: jax.random.PRNGKey, num_points: int = 10,
                                   normalize: bool = True) -> jnp.ndarray:
        x_domain = jax.random.uniform(key, shape=(num_points, self.input_size),
                                      minval=self.domain_l, maxval=self.domain_u)
        if normalize:
            x_domain = self._normalize_data(x_domain)
        assert x_domain.shape == (num_points, self.input_size)
        return x_domain

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_kernel(self, pred_raw: jnp.ndarray):
        assert pred_raw.ndim == 3 and pred_raw.shape[-1] == self.output_size
        pred_raw = pred_raw.reshape((pred_raw.shape[0], -1))
        particles_copy = jax.lax.stop_gradient(pred_raw)
        k = self.kernel_svgd.matrix(pred_raw, particles_copy)
        return jnp.sum(k), k

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        self.opt_state, self.params_stack, stats = self._step_jit(self.opt_state, self.params_stack, x_batch, y_batch,
                                                                  key=self.rng_key, num_train_points=num_train_points)
        return stats

    @partial(jax.jit, static_argnums=(0,))
    def _surrogate_loss(self, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                        num_train_points: int, key: jax.random.PRNGKey) -> [jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key, 2)

        # combine the training data batch with a batch of sampled measurement points
        train_batch_size = x_batch.shape[0]
        x_domain = self._sample_measurement_points(key1, num_points=self.num_measurement_points)
        x_stacked = jnp.concatenate([x_batch, x_domain], axis=0)

        # likelihood
        f_raw = self.batched_model.forward_vec(x_stacked, param_vec_stack)
        (_, post_stats), grad_post = jax.value_and_grad(self._neg_log_posterior_surrogate, has_aux=True)(
            f_raw, x_stacked, y_batch, train_batch_size, num_train_points, key2)

        # kernel
        grad_k, k = jax.grad(self._evaluate_kernel, has_aux=True)(f_raw)

        surrogate_loss = jnp.sum(f_raw * jax.lax.stop_gradient(jnp.einsum('ij,jkm', k, grad_post)
                                                               + grad_k / self.num_particles))
        # surrogate_loss = jnp.sum(f_raw * jax.lax.stop_gradient(jnp.einsum('ij,jkm', k, grad_k)
        #                                                        + grad_post / self.num_particles))
        avg_triu_k = jnp.sum(jnp.triu(k, k=1)) / ((self.num_particles - 1) * self.num_particles / 2)
        stats = OrderedDict(**post_stats, avg_triu_k=avg_triu_k)
        return surrogate_loss, stats

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, opt_state: optax.OptState, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                  key: jax.random.PRNGKey, num_train_points: Union[float, int]):

        (loss, stats), grad = jax.value_and_grad(self._surrogate_loss, has_aux=True)(
            param_vec_stack, x_batch, y_batch, num_train_points, key)
        updates, opt_state = self.optim.update(grad, opt_state, param_vec_stack)
        param_vec_stack = optax.apply_updates(param_vec_stack, updates)
        return opt_state, param_vec_stack, stats

    def _nll(self, pred_raw: jnp.ndarray, y_batch: jnp.ndarray, train_data_till_idx: int):
        likelihood_std = self.likelihood_std
        log_prob = tfd.MultivariateNormalDiag(pred_raw[:, :train_data_till_idx, :], likelihood_std).log_prob(y_batch)
        return - jnp.mean(log_prob)

    def _neg_log_posterior_surrogate(self, pred_raw: jnp.ndarray, x_stacked: jnp.ndarray, y_batch: jnp.ndarray,
                                     train_data_till_idx: int, num_train_points: Union[float, int],
                                     key: jax.random.PRNGKey):
        nll = self._nll(pred_raw, y_batch, train_data_till_idx)
        prior_score = self._estimate_prior_score(x_stacked, pred_raw, key) / num_train_points
        neg_log_post = nll - jnp.sum(jnp.mean(pred_raw * jax.lax.stop_gradient(prior_score), axis=-2))
        stats = OrderedDict(train_nll_loss=nll)
        return neg_log_post, stats

    def _estimate_prior_score(self, x: jnp.array, y: jnp.array, key: jax.random.PRNGKey) -> jnp.ndarray:
        x_unnormalized = self._unnormalize_data(x)
        f_prior = self.function_sim.sample_function_vals(x=x_unnormalized, num_samples=self.num_f_samples, rng_key=key)
        f_prior_normalized = self._normalize_y(f_prior)
        if self.independent_output_dims:
            # performs score estimation for each output dimension independently
            ssge_score = self.estimate_gradients_s_x_vectorized(y, f_prior_normalized)
        else:
            # performs score estimation for all output dimensions jointly
            # befor score estimation call, flatten the output dimensions
            ssge_score = self.ssge.estimate_gradients_s_x(y.reshape((y.shape[0], -1)),
                                                          f_prior_normalized.reshape((f_prior_normalized.shape[0], -1)))
            # add back the output dimensions
            ssge_score = ssge_score.reshape(y.shape)
        assert ssge_score.shape == y.shape
        return ssge_score

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True) -> tfd.Distribution:
        self.batched_model.param_vectors_stacked = self.params_stack
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        pred_dist = self._to_pred_dist(y_pred_raw, likelihood_std=self.likelihood_std, include_noise=include_noise)
        assert pred_dist.batch_shape == x.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        return pred_dist

    def predict_post_samples(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        y_pred = y_pred_raw * self._y_std + self._y_mean
        assert y_pred.ndim == 3 and y_pred.shape[-2:] == (x.shape[0], self.output_size)
        return y_pred


if __name__ == '__main__':
    from sim_transfer.sims import GaussianProcessSim, SinusoidsSim

    def key_iter():
        key = jax.random.PRNGKey(7644)
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

    domain_l, domain_u = np.array([-7.] * NUM_DIM_X), np.array([7.] * NUM_DIM_X)

    x_train = jax.random.uniform(next(key_iter), shape=(num_train_points, NUM_DIM_X), minval=-5, maxval=5)
    y_train = fun(x_train) + 0.1 * jax.random.normal(next(key_iter), shape=(x_train.shape[0], NUM_DIM_Y))

    num_test_points = 100
    x_test = jax.random.uniform(next(key_iter), shape=(num_test_points, NUM_DIM_X), minval=-5, maxval=5)
    y_test = fun(x_test) + 0.1 * jax.random.normal(next(key_iter), shape=(x_test.shape[0], NUM_DIM_Y))


    # sim = GaussianProcessSim(input_size=1, output_scale=3.0, mean_fn=lambda x: 2 * x)
    sim = SinusoidsSim(input_size=1, output_size=NUM_DIM_Y)
    bnn = BNN_FSVGD_Sim_Prior(NUM_DIM_X, NUM_DIM_Y, domain_l, domain_u, rng_key=next(key_iter), function_sim=sim,
                              hidden_layer_sizes=[64, 64, 64],
                              num_train_steps=20000, data_batch_size=4,
                              independent_output_dims=True)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
        if NUM_DIM_X == 1:
            bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 5000}',
                        domain_l=-7, domain_u=7)
