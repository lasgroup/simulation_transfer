from collections import OrderedDict
from functools import partial
from typing import List, Optional, Callable, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np

from sim_transfer.models.bnn import AbstractFSVGD_BNN
from sim_transfer.score_estimation import SSGE
from sim_transfer.sims import FunctionSimulator, Domain, HypercubeDomain


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
                 bandwidth_ssge: float = 1.,
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

        # initialize ssge algo
        self.ssge = SSGE(bandwidth=bandwidth_ssge)

        # create estimate gradient function over the output dims
        self.independent_output_dims = independent_output_dims
        if independent_output_dims:
            self.estimate_gradients_s_x_vectorized = jax.vmap(lambda y, f: self.ssge.estimate_gradients_s_x(y, f),
                                                              in_axes=-1,
                                                              out_axes=-1)

    @partial(jax.jit, static_argnums=(0,))
    def _surrogate_loss(self, params: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                        num_train_points: int, key: jax.random.PRNGKey) -> [jnp.ndarray, Dict]:
        key1, key2 = jax.random.split(key, 2)

        # combine the training data batch with a batch of sampled measurement points
        train_batch_size = x_batch.shape[0]
        x_domain = self._sample_measurement_points(key1, num_points=self.num_measurement_points)
        x_stacked = jnp.concatenate([x_batch, x_domain], axis=0)

        # get likelihood std
        if self.learn_likelihood_std:
            likelihood_std = self._likelihood_std_transform(params['likelihood_std_raw'])
        else:
            likelihood_std = self.likelihood_std

        # posterior score
        f_raw = self.batched_model.forward_vec(x_stacked, params['nn_params_stacked'])
        (_, post_stats), (grad_post_f, grad_post_lstd) = jax.value_and_grad(
            self._neg_log_posterior_surrogate, argnums=[0, 1], has_aux=True)(
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

    def _neg_log_posterior_surrogate(self, pred_raw: jnp.ndarray, likelihood_std: jnp.array, x_stacked: jnp.ndarray,
                                     y_batch: jnp.ndarray, train_data_till_idx: int,
                                     num_train_points: Union[float, int], key: jax.random.PRNGKey):
        nll = - self._ll(pred_raw, likelihood_std, y_batch, train_data_till_idx)
        prior_score = self._estimate_prior_score(x_stacked, pred_raw, key) / num_train_points
        neg_log_post = nll - jnp.sum(jnp.mean(pred_raw * jax.lax.stop_gradient(prior_score), axis=-2))
        stats = OrderedDict(train_nll_loss=nll)
        if self.learn_likelihood_std:
            stats['likelihood_std'] = jnp.mean(likelihood_std)
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


if __name__ == '__main__':
    from sim_transfer.sims import SinusoidsSim, QuadraticSim


    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key


    key_iter = key_iter()
    NUM_DIM_X = 1
    NUM_DIM_Y = 1
    SIM_TYPE = 'SinusoidsSim'

    if SIM_TYPE == 'QuadraticSim':
        sim = QuadraticSim()
        fun = lambda x: (x - 2) ** 2
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

    bnn = BNN_FSVGD_SimPrior(NUM_DIM_X, NUM_DIM_Y, domain=domain, rng_key=next(key_iter), function_sim=sim,
                             hidden_layer_sizes=[64, 64, 64], num_train_steps=20000, data_batch_size=4,
                             learn_likelihood_std=False, num_f_samples=64, bandwidth_svgd=0.05, bandwidth_ssge=0.2,
                             normalization_stats=sim.normalization_stats)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
        if NUM_DIM_X == 1:
            bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 5000}',
                        domain_l=domain.l[0], domain_u=domain.u[0])
