from typing import Union, Dict, Union
from functools import partial
from collections import OrderedDict
import jax.numpy as jnp
import jax
import optax

import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates import jax as tfp

from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel

class AbstractParticleBNN(BatchedNeuralNetworkModel):

    def __init__(self, likelihood_std: Union[float, jnp.array] = 0.2, lr: float = 1e-3, **kwargs):
        super().__init__(**kwargs)

        # setup likelihood std
        if isinstance(likelihood_std, float):
            self.likelihood_std = likelihood_std * jnp.ones(self.output_size)
        elif isinstance(likelihood_std, jnp.ndarray):
            assert likelihood_std.shape == (self.output_size,)
            self.likelihood_std = likelihood_std
        else:
            raise ValueError(f'likelihood_std must be float or jnp.ndarray of size ({self.output_size},)')

        # initialize batched NN
        self.params_stack = self.batched_model.param_vectors_stacked

        # initialize optimizer
        self.optim = optax.adam(learning_rate=lr)
        self.opt_state = self.optim.init(self.params_stack)

    def _surrogate_loss(self, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                        num_train_points: int, key: jax.random.PRNGKey) -> [jnp.ndarray, Dict]:
        raise NotImplementedError('Needs to be implemented by subclass')

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, opt_state: optax.OptState, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                  key: jax.random.PRNGKey, num_train_points: Union[float, int]):
        (loss, stats), grad = jax.value_and_grad(self._surrogate_loss, has_aux=True)(
            param_vec_stack, x_batch, y_batch, num_train_points, key)
        updates, opt_state = self.optim.update(grad, opt_state, param_vec_stack)
        param_vec_stack = optax.apply_updates(param_vec_stack, updates)
        return opt_state, param_vec_stack, stats

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        self.opt_state, self.params_stack, stats = self._step_jit(self.opt_state, self.params_stack, x_batch, y_batch,
                                                                  key=self.rng_key, num_train_points=num_train_points)
        return stats


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


class MeasurementSetMixin:
    def __init__(self, domain_l: jnp.ndarray, domain_u: jnp.ndarray):
        assert isinstance(self, AbstractParticleBNN)

        # check and set domain boundaries
        assert domain_u.shape == domain_l.shape == (self.input_size,)
        assert jnp.all(domain_l <= domain_u), 'lower bound of domain must be smaller than upper bound'
        self.domain_l = domain_l
        self.domain_u = domain_u

    def _sample_measurement_points(self, key: jax.random.PRNGKey, num_points: int = 10,
                                   normalize: bool = True) -> jnp.ndarray:
        """ Samples measurement points from the domain """
        x_domain = jax.random.uniform(key, shape=(num_points, self.input_size),
                                      minval=self.domain_l, maxval=self.domain_u)
        if normalize:
            x_domain = self._normalize_data(x_domain)
        assert x_domain.shape == (num_points, self.input_size)
        return x_domain

    def _nll(self, pred_raw: jnp.ndarray, y_batch: jnp.ndarray, train_data_till_idx: int):
        likelihood_std = self.likelihood_std
        log_prob = tfd.MultivariateNormalDiag(pred_raw[:, :train_data_till_idx, :], likelihood_std).log_prob(y_batch)
        return - jnp.mean(log_prob)


class AbstractFSVGD_BNN(AbstractParticleBNN, MeasurementSetMixin):

    def __init__(self, domain_l: jnp.ndarray, domain_u: jnp.ndarray, bandwidth_svgd: float = 0.4, **kwargs):
        AbstractParticleBNN.__init__(self, **kwargs)
        MeasurementSetMixin.__init__(self, domain_l, domain_u)
        self.bandwidth_svgd = bandwidth_svgd
        self.kernel_svgd = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_svgd)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_kernel(self, pred_raw: jnp.ndarray):
        assert pred_raw.ndim == 3 and pred_raw.shape[-1] == self.output_size
        pred_raw = pred_raw.reshape((pred_raw.shape[0], -1))
        particles_copy = jax.lax.stop_gradient(pred_raw)
        k = self.kernel_svgd.matrix(pred_raw, particles_copy)
        return jnp.sum(k), k

    @property
    def num_particles(self) -> int:
        return self.num_batched_nns

class AbstractSVGD_BNN(AbstractParticleBNN):


    def __init__(self, bandwidth_svgd: float = 0.4, use_prior: bool = True, **kwargs):
        AbstractParticleBNN.__init__(self, **kwargs)
        self.use_prior = use_prior
        self.bandwidth_svgd = bandwidth_svgd
        self.svgd_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_svgd)

    @property
    def prior_dist(self) -> tfd.Distribution:
        raise NotImplementedError('Needs to be implemented by subclass')
    @property
    def num_particles(self) -> int:
        return self.num_batched_nns

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_kernel(self, particles: jnp.ndarray):
        particles_copy = jax.lax.stop_gradient(particles)
        k = self.svgd_kernel.matrix(particles, particles_copy)
        return jnp.sum(k), k

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        self.opt_state, self.params_stack, stats = self._step_jit(self.opt_state, self.params_stack, x_batch, y_batch,
                                                                  num_train_points)
        return stats

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, opt_state: optax.OptState, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                  num_train_points: Union[float, int]):
        # SVGD updates
        (log_post, post_stats), grad_q = jax.value_and_grad(self._neg_log_posterior, has_aux=True)(param_vec_stack,
                                                                                                   x_batch, y_batch,
                                                                                                   num_train_points)
        grad_q = self.batched_model.flatten_batch(grad_q)

        grad_k, k = jax.grad(self._evaluate_kernel, has_aux=True)(self.batched_model.flatten_batch(param_vec_stack))
        grad = k @ grad_q + grad_k / self.num_particles

        updates, opt_state = self.optim.update(self.batched_model.unravel_batch(grad), opt_state, param_vec_stack)
        param_vec_stack = optax.apply_updates(param_vec_stack, updates)

        avg_triu_k = jnp.sum(jnp.triu(k, k=1)) / ((self.num_particles - 1) * self.num_particles / 2)
        stats = OrderedDict(**post_stats, avg_grad_q=jnp.mean(grad_q), avg_grad_k=jnp.mean(grad_q),
                            avg_triu_k=avg_triu_k)

        return opt_state, param_vec_stack, stats

    def _ll(self, param_vec_stack: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
        pred_raw = self.batched_model.forward_vec(x_batch, param_vec_stack)
        log_prob = tfd.MultivariateNormalDiag(pred_raw, self.likelihood_std).log_prob(y_batch)
        return jnp.mean(log_prob)

    def _neg_log_posterior(self, param_vec_stack: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray,
                           num_train_points: Union[float, int]):
        ll = self._ll(param_vec_stack, x_batch=x_batch, y_batch=y_batch)
        if self.use_prior:
            log_prior = jnp.mean(self.prior_dist.log_prob(self.batched_model.flatten_batch(param_vec_stack)))
            log_prior /= (num_train_points * self.prior_dist.event_shape[0])
            stats = OrderedDict(train_nll_loss=-ll, neg_log_prior=-log_prior)
            log_posterior = ll + log_prior
        else:
            log_posterior = ll
            stats = OrderedDict(train_nll_loss=-ll)
        return - log_posterior, stats