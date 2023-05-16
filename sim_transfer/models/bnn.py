from typing import Optional, Dict, Union
from functools import partial
from collections import OrderedDict
from jaxtyping import PyTree
import jax.numpy as jnp
import jax
import optax

import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates import jax as tfp
from sim_transfer.sims import Domain

from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel


class LikelihoodMixin:

    def __init__(self, likelihood_std: Union[float, jnp.array] = 0.2,
                 learn_likelihood_std: bool = False,
                 normalize_likelihood_std: bool = False):
        self.learn_likelihood_std = learn_likelihood_std

        if normalize_likelihood_std:
            assert hasattr(self, '_y_std') and self._y_std is not None and self.normalize_data, \
                'normalize_likelihood_std requires normalization'
            assert self._y_std.shape == (self.output_size, )
            self.likelihood_std = likelihood_std / self._y_std
        else:
            self.likelihood_std = likelihood_std


        assert hasattr(self, 'params'), 'super class must have params attribute'

    @property
    def likelihood_std(self) -> jnp.ndarray:
        if self.learn_likelihood_std:
            return self._likelihood_std_transform(self.params['likelihood_std_raw'])
        else:
            return self._likelihood_std

    @likelihood_std.setter
    def likelihood_std(self, std_value: Union[float, jnp.ndarray]):
        if isinstance(std_value, float):
            _likelihood_std = std_value * jnp.ones(self.output_size)
        elif isinstance(std_value, jnp.ndarray):
            assert std_value.shape == (self.output_size,)
            _likelihood_std = std_value
        else:
            raise ValueError(f'likelihood_std must be float or jnp.ndarray of size ({self.output_size},)')
        assert jnp.all(_likelihood_std > 0), 'likelihood_std must be positive'

        if self.learn_likelihood_std:
            self.params['likelihood_std_raw'] = self._likelihood_std_transform_inv(_likelihood_std)
        else:
            self._likelihood_std = _likelihood_std

    def _likelihood_std_transform(self, std_value: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.softplus(std_value)

    def _likelihood_std_transform_inv(self, std_value: jnp.ndarray) -> jnp.ndarray:
        return tfp.math.softplus_inverse(std_value)


class AbstractParticleBNN(BatchedNeuralNetworkModel, LikelihoodMixin):

    def __init__(self, likelihood_std: Union[float, jnp.array] = 0.2, learn_likelihood_std: bool = False,
                 lr: float = 1e-3, weight_decay: float = 0.0, normalize_likelihood_std: bool = False, **kwargs):
        self.params = {}  # this must happen before super().__init__ is called
        BatchedNeuralNetworkModel.__init__(self, **kwargs)
        LikelihoodMixin.__init__(self, likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std,
                                 normalize_likelihood_std=normalize_likelihood_std)

        # initialize batched NN
        self.params.update({'nn_params_stacked': self.batched_model.param_vectors_stacked})

        # initialize optimizer
        if weight_decay > 0:
            self.optim = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
        else:
            self.optim = optax.adam(learning_rate=lr)
        self.opt_state = self.optim.init(self.params)


    def _surrogate_loss(self, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                        num_train_points: int, key: jax.random.PRNGKey) -> [jnp.ndarray, Dict]:
        raise NotImplementedError('Needs to be implemented by subclass')

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def _step(self, opt_state: optax.OptState, params: Dict, x_batch: jnp.array, y_batch: jnp.array,
                  key: jax.random.PRNGKey, num_train_points: Union[float, int]):
        (loss, stats), grad = jax.value_and_grad(self._surrogate_loss, has_aux=True)(
            params, x_batch, y_batch, num_train_points, key)
        updates, opt_state = self.optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, stats

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        self.opt_state, self.params, stats = self._step_jit(self.opt_state, self.params, x_batch, y_batch,
                                                            key=self.rng_key, num_train_points=num_train_points)
        return stats

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True) -> tfd.Distribution:
        self.batched_model.param_vectors_stacked = self.params['nn_params_stacked']
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


class AbstractVariationalBNN(BatchedNeuralNetworkModel, LikelihoodMixin):

    def __init__(self, likelihood_std: Union[float, jnp.array] = 0.2, learn_likelihood_std: bool = False,
                 normalize_likelihood_std: bool = False, **kwargs):
        self.params = {} # this must happen before super().__init__ is called
        BatchedNeuralNetworkModel.__init__(self, **kwargs)
        LikelihoodMixin.__init__(self, likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std,
                                 normalize_likelihood_std=normalize_likelihood_std)

        # need to be implemented by subclass
        self.optim = None
        self.opt_state = None

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        self.opt_state, self.params, stats = self._step_jit(self.opt_state, self.params, x_batch, y_batch,
                                                            key=self.rng_key, num_train_points=num_train_points)
        return stats

    def _loss(self, params: Dict, x_batch: jnp.array, y_batch: jnp.array,
                    num_train_points: int, key: jax.random.PRNGKey) -> [jnp.ndarray, Dict]:
        raise NotImplementedError('Needs to be implemented by subclass')

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, opt_state: optax.OptState, params: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                  key: jax.random.PRNGKey, num_train_points: Union[float, int]):
        (loss, stats), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params, x_batch, y_batch, num_train_points, key)
        updates, opt_state = self.optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, stats

    def _post_predict_raw(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None,
                     num_post_samples: Optional[int] = None) -> jnp.ndarray:
        if key is None:
            key = self.rng_key
        if num_post_samples is None:
            num_post_samples = self.num_post_samples
        params_stack = self.posterior_dist.sample(sample_shape=num_post_samples, seed=key)
        self.batched_model.param_vectors_stacked = self.batched_model.unravel_batch(params_stack)
        y_pred_raw = self.batched_model(x)
        return y_pred_raw

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True,
                     key: Optional[jax.random.PRNGKey] = None,
                     num_post_samples: Optional[int] = None) -> tfd.Distribution:
        # normalize input data
        x = self._normalize_data(x)

        # sample NNs from posterior and return the corresponding (raw) predictions
        y_pred_raw = self._post_predict_raw(x, key=key, num_post_samples=num_post_samples)

        # convert sampled NN predictions into predictive distribution.
        pred_dist = self._to_pred_dist(y_pred_raw, likelihood_std=self.likelihood_std, include_noise=include_noise)

        # do shape checks
        assert pred_dist.batch_shape == x.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        return pred_dist

    def predict_post_samples(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None,
                     num_post_samples: Optional[int] = None) -> jnp.ndarray:
        # normalize input data
        x = self._normalize_data(x)

        # sample NNs from posterior and return the corresponding (raw) predictions
        y_pred_raw = self._post_predict_raw(x, key=key, num_post_samples=num_post_samples)

        # denormalize output data
        y_pred = y_pred_raw * self._y_std + self._y_mean

        # do shape checks
        assert y_pred.ndim == 3 and y_pred.shape[-2:] == (x.shape[0], self.output_size)
        if num_post_samples is None:
            num_post_samples = self.num_post_samples
        assert y_pred.shape[0] == num_post_samples
        return y_pred

    @property
    def posterior_dist(self) -> tfd.Distribution:
        raise NotImplementedError

    @property
    def prior_dist(self) -> tfd.Distribution:
        raise NotImplementedError

    @property
    def num_post_samples(self) -> int:
        return self.num_batched_nns


class MeasurementSetMixin:
    def __init__(self, domain: Domain):
        assert isinstance(self, AbstractParticleBNN)
        assert isinstance(domain, Domain)
        # check and set domain boundaries
        assert domain.num_dims == self.input_size
        self.domain = domain

    def _sample_measurement_points(self, key: jax.random.PRNGKey, num_points: int = 10,
                                   normalize: bool = True) -> jnp.ndarray:
        """ Samples measurement points from the domain """
        x_samples = self.domain.sample_uniformly(key, sample_shape=(num_points,))
        if normalize:
            x_samples = self._normalize_data(x_samples)
        assert x_samples.shape == (num_points, self.input_size)
        return x_samples

    def _ll(self, pred_raw: jnp.array, likelihood_std: jnp.array, y_batch: jnp.ndarray, train_data_till_idx: int):
        log_prob = tfd.MultivariateNormalDiag(pred_raw[:, :train_data_till_idx, :], likelihood_std).log_prob(y_batch)
        return jnp.sum(jnp.mean(log_prob, axis=-1), axis=0)  # take mean over batch and sum over NNs


class AbstractFSVGD_BNN(AbstractParticleBNN, MeasurementSetMixin):

    def __init__(self, domain, bandwidth_svgd: float = 0.4, **kwargs):
        AbstractParticleBNN.__init__(self, **kwargs)
        MeasurementSetMixin.__init__(self, domain=domain)
        self.bandwidth_svgd = bandwidth_svgd
        self.kernel_svgd = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_svgd)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_kernel(self, pred_raw: jnp.ndarray):
        assert pred_raw.ndim == 3 and pred_raw.shape[-1] == self.output_size
        pred_raw = pred_raw.reshape((pred_raw.shape[0], -1))
        particles_copy = jax.lax.stop_gradient(pred_raw)
        k = self.kernel_svgd.matrix(pred_raw, particles_copy)
        return jnp.sum(k), k

    def _neg_log_posterior(self, pred_raw: jnp.ndarray, likelihood_std: jnp.array, x_stacked: jnp.ndarray,
                           y_batch: jnp.ndarray, train_data_till_idx: int,
                           num_train_points: Union[float, int], key: jax.random.PRNGKey):
        raise NotImplementedError

    def _surrogate_loss(self, params: Dict, x_batch: jnp.array, y_batch: jnp.array,
                        num_train_points: int, key: jax.random.PRNGKey) -> [jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key, 2)

        # combine the training data batch with a batch of sampled measurement points
        train_batch_size = x_batch.shape[0]
        x_domain = self._sample_measurement_points(key1, num_points=self.num_measurement_points)
        x_stacked = jnp.concatenate([x_batch, x_domain], axis=0)

        # get likelihood std
        likelihood_std = self._likelihood_std_transform(params['likelihood_std_raw']) if self.learn_likelihood_std \
            else self.likelihood_std

        # posterior score
        f_raw = self.batched_model.forward_vec(x_stacked, params['nn_params_stacked'])
        (_, post_stats), (grad_post_f, grad_post_lstd) = jax.value_and_grad(
            self._neg_log_posterior, argnums=[0, 1], has_aux=True)(
            f_raw, likelihood_std, x_stacked, y_batch, train_batch_size, num_train_points, key2)

        # kernel
        grad_k, k = jax.grad(self._evaluate_kernel, has_aux=True)(f_raw)

        # construct surrogate loss such that the gradient of the surrogate loss is the fsvgd update
        surrogate_loss = jnp.sum(f_raw * jax.lax.stop_gradient(jnp.einsum('ij,jkm', k, grad_post_f)
                                                               + grad_k / self.num_particles))
        if self.learn_likelihood_std:
            surrogate_loss += jnp.sum(likelihood_std * jax.lax.stop_gradient(grad_post_lstd))
        avg_triu_k = jnp.sum(jnp.triu(k, k=1)) / ((self.num_particles - 1) * self.num_particles / 2)
        stats = OrderedDict(**post_stats, avg_triu_k=avg_triu_k)
        return surrogate_loss, stats

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
        self.opt_state, self.params, stats = self._step_jit(self.opt_state, self.params, x_batch, y_batch,
                                                            num_train_points)
        return stats

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, opt_state: optax.OptState, params: PyTree, x_batch: jnp.array, y_batch: jnp.array,
                  num_train_points: Union[float, int]):
        # SVGD updates
        (log_post, post_stats), grad_params_q = jax.value_and_grad(self._neg_log_posterior, has_aux=True)(
            params, x_batch, y_batch, num_train_points)
        grad_q = self.batched_model.flatten_batch(grad_params_q['nn_params_stacked'])

        nn_params_vec_stack = self.batched_model.flatten_batch(params['nn_params_stacked'])
        grad_k, k = jax.grad(self._evaluate_kernel, has_aux=True)(nn_params_vec_stack)
        grad = k @ grad_q + grad_k / self.num_particles

        grad_params_q['nn_params_stacked'] = self.batched_model.unravel_batch(grad)
        updates, opt_state = self.optim.update(grad_params_q, opt_state, params)
        params = optax.apply_updates(params, updates)

        avg_triu_k = jnp.sum(jnp.triu(k, k=1)) / ((self.num_particles - 1) * self.num_particles / 2)
        stats = OrderedDict(**post_stats, avg_grad_q=jnp.mean(grad_q), avg_grad_k=jnp.mean(grad_q),
                            avg_triu_k=avg_triu_k)
        return opt_state, params, stats

    def _ll(self, params: Dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
        pred_raw = self.batched_model.forward_vec(x_batch, params['nn_params_stacked'])
        if self.learn_likelihood_std:
            likelihood_std = self._likelihood_std_transform(params['likelihood_std_raw'])
        else:
            likelihood_std = self.likelihood_std
        log_prob = tfd.MultivariateNormalDiag(pred_raw, likelihood_std).log_prob(y_batch)
        return jnp.mean(log_prob)

    def _neg_log_posterior(self, params: Dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray,
                           num_train_points: Union[float, int]):
        ll = self._ll(params, x_batch=x_batch, y_batch=y_batch)
        if self.use_prior:
            log_prior = jnp.mean(self.prior_dist.log_prob(
                self.batched_model.flatten_batch(params['nn_params_stacked'])))
            log_prior /= (num_train_points * self.prior_dist.event_shape[0])
            stats = OrderedDict(train_nll_loss=-ll, neg_log_prior=-log_prior)
            log_posterior = ll + log_prior
        else:
            log_posterior = ll
            stats = OrderedDict(train_nll_loss=-ll)
        if self.learn_likelihood_std:
            stats['likelihood_std'] = jnp.mean(self._likelihood_std_transform(params['likelihood_std_raw']))
        return - log_posterior, stats