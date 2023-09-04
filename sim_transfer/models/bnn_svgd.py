from typing import List, Optional, Callable, Dict, Union

import jax
import jax.numpy as jnp

from sim_transfer.models.bnn import AbstractSVGD_BNN


class BNN_SVGD(AbstractSVGD_BNN):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 rng_key: jax.random.PRNGKey,
                 likelihood_std: Union[float, jnp.array] = 0.2,
                 learn_likelihood_std: bool = False,
                 likelihood_exponent: float = 1.0,
                 num_particles: int = 10,
                 bandwidth_svgd: float = 10.0,
                 data_batch_size: int = 16,
                 num_train_steps: int = 10000,
                 lr=1e-3,
                 weight_decay: float = 1e-3,
                 normalize_data: bool = True,
                 normalize_likelihood_std: bool = False,
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
                         normalize_likelihood_std=normalize_likelihood_std,
                         lr=lr, weight_decay=weight_decay, likelihood_std=likelihood_std,
                         learn_likelihood_std=learn_likelihood_std,
                         likelihood_exponent=likelihood_exponent,
                         bandwidth_svgd=bandwidth_svgd, use_prior=use_prior)
        self._save_init_args(locals())

        # construct the neural network prior distribution
        if use_prior:
            self._prior_dist = self._construct_nn_param_prior(weight_prior_std, bias_prior_std)

    @property
    def prior_dist(self):
        return self._prior_dist

    def _get_state(self):
        state_dict = {
            'opt_state': self.opt_state,
            'params': self.params,
            '_rng_key': self._rng_key,
            '_x_mean': self._x_mean,
            '_x_std': self._x_std,
            '_y_mean': self._y_mean,
            '_y_std': self._y_std,
            'affine_transform_y': self.affine_transform_y
        }
        return state_dict


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

    bnn = BNN_SVGD(NUM_DIM_X, NUM_DIM_Y, next(key_iter), num_train_steps=20000, bandwidth_svgd=10., likelihood_std=0.05,
                   likelihood_exponent=1.0)
    # bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=20000)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
        bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'SVGD, iter {(i + 1) * 2000}',
                    domain_l=domain.l[0], domain_u=domain.u[0])
