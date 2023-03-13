from collections import OrderedDict
from functools import partial
from typing import List, Optional, Callable, Dict, Union

import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates import jax as tfp

from sim_transfer.models.bnn import AbstractSVGD_BNN


class BNN_SVGD(AbstractSVGD_BNN):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 rng_key: jax.random.PRNGKey,
                 likelihood_std: Union[float, jnp.array] = 0.2,
                 learn_likelihood_std: bool = False,
                 num_particles: int = 10,
                 bandwidth_svgd: float = 10.0,
                 data_batch_size: int = 16,
                 num_train_steps: int = 10000,
                 lr=1e-3,
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
                         num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=normalization_stats,
                         lr=lr, likelihood_std=likelihood_std, learn_likelihood_std=learn_likelihood_std,
                         bandwidth_svgd=bandwidth_svgd, use_prior=use_prior)

        # construct the neural network prior distribution
        if use_prior:
            self._prior_dist = self._construct_nn_param_prior(weight_prior_std, bias_prior_std)

    @property
    def prior_dist(self):
        return self._prior_dist


if __name__ == '__main__':
    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key


    key_iter = key_iter()

    fun = lambda x: 2 * x + 2 * jnp.sin(2 * x)

    num_train_points = 10
    x_train = jax.random.uniform(next(key_iter), shape=(num_train_points, 1), minval=-5, maxval=5)
    y_train = fun(x_train) + 0.1 * jax.random.normal(next(key_iter), shape=x_train.shape)

    num_test_points = 100
    x_test = jax.random.uniform(next(key_iter), shape=(num_test_points, 1), minval=-5, maxval=5)
    y_test = fun(x_test) + 0.1 * jax.random.normal(next(key_iter), shape=x_test.shape)

    bnn = BNN_SVGD(1, 1, next(key_iter), num_train_steps=20000, bandwidth_svgd=0.2, data_batch_size=50, use_prior=True,
                   learn_likelihood_std=True, likelihood_std=0.1)
    # bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=20000)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
        bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 2000}')
