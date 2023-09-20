from sim_transfer.models import BNN_SVGD
import jax.numpy as jnp
from typing import Optional, Dict
import jax
from sim_transfer.sims.dynamics_models import RaceCar, CarParams
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from sim_transfer.modules.distribution import AffineTransform


class GreyBoxSVGDCarModel(BNN_SVGD):

    def __init__(self, encode_angle: bool = True, high_fidelity: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.car_model_params = {
            'i_com': jnp.array(27.8e-6),
            'd_f': jnp.array(0.02),
            'c_f': jnp.array(1.2),
            'b_f': jnp.array(2.58),
            'd_r': jnp.array(0.017),
            'c_r': jnp.array(1.27),
            'b_r': jnp.array(3.39),
            'c_m_1': jnp.array(10.0),
            'c_m_2': jnp.array(0.05),
            'c_d': jnp.array(0.52),
            'steering_limit': jnp.array(0.35),
            'blend_ratio_ub': jnp.array([0.005]),
            'blend_ratio_lb': jnp.array([0.004]),
            'angle_offset': jnp.array([0.0]),
        }
        self.params.update({'car_model_params': self.car_model_params})
        self._init_optim()
        self.dynamics = RaceCar(dt=1 / 30., encode_angle=encode_angle, rk_integrator=True)
        self.state_dim = 6 + int(encode_angle)
        self.high_fidelity = high_fidelity

    def reinit(self, rng_key: Optional[jax.random.PRNGKey] = None):
        """ Reinitializes the model parameters and the optimizer state."""
        if rng_key is None:
            rng_key = self.rng_key
        key_model, key_rng = jax.random.split(rng_key)
        self._rng_key = key_rng  # reinitialize rng_key
        self.batched_model.reinit_params(key_model)  # reinitialize model parameters
        self.params['nn_params_stacked'] = self.batched_model.param_vectors_stacked
        self.params['car_model_params'] = self.car_model_params
        self._init_likelihood()  # reinitialize likelihood std
        self._init_optim()  # reinitialize optimizer

    def sim_model_step(self, x: jnp.array, car_params: CarParams):
        state, u = x[..., :self.state_dim], x[..., -2:]
        model_params = CarParams(**car_params, m=1.65, g=9.81,
                                 use_blend=int(self.high_fidelity),
                                 l_f=0.13, l_r=0.17)
        return jax.vmap(self.dynamics.next_step, in_axes=(0, 0, None), out_axes=0)(
            state, u, model_params)

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True):
        pred_dist_bnn = super().predict_dist(x, include_noise)
        sim_model_prediction = self.sim_model_step(x, car_params=self.params['car_model_params'])
        affine_transform_y = AffineTransform(shift=sim_model_prediction, scale=1.0)
        pred_dist = affine_transform_y(pred_dist_bnn)
        assert pred_dist.batch_shape == x.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        return pred_dist

    def predict_post_samples(self, x: jnp.ndarray) -> jnp.ndarray:
        sim_model_prediction = self.sim_model_step(x, car_params=self.params['car_model_params'])
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        y_pred = y_pred_raw * self._y_std + self._y_mean
        y_pred = jax.tree_util.tree_map(lambda y: y + sim_model_prediction, y_pred)
        assert y_pred.ndim == 3 and y_pred.shape[-2:] == (x.shape[0], self.output_size)
        return y_pred

    def _ll(self, params: Dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
        """ computes the avg log likelihood of the batch """
        pred_raw = self.batched_model.forward_vec(x_batch, params['nn_params_stacked'])
        if self.learn_likelihood_std:
            likelihood_std = self._likelihood_std_transform(params['likelihood_std_raw'])
        else:
            likelihood_std = self.likelihood_std
        unormalized_x = self._unnormalize_data(x_batch)
        sim_model_prediction = self.sim_model_step(unormalized_x, car_params=params['car_model_params'])
        normalized_sim_model_predictions = self._normalize_y(sim_model_prediction)
        log_prob = tfd.MultivariateNormalDiag(pred_raw + normalized_sim_model_predictions,
                                              likelihood_std).log_prob(y_batch)
        return jnp.mean(log_prob)


if __name__ == '__main__':
    from sim_transfer.sims.simulators import RaceCarSim

    sim = RaceCarSim(encode_angle=True, use_blend=True)


    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key


    key_iter = key_iter()
    NUM_DIM_X = 9
    NUM_DIM_Y = 7
    x_train, y_train, x_test, y_test = sim.sample_datasets(rng_key=next(key_iter), num_samples_train=20000,
                                                           obs_noise_std=0.05)
    bnn = GreyBoxSVGDCarModel(True, True, NUM_DIM_X, NUM_DIM_Y, next(key_iter), num_train_steps=20000,
                              bandwidth_svgd=10., likelihood_std=0.05, likelihood_exponent=1.0,
                              normalize_likelihood_std=True, learn_likelihood_std=False)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000, per_dim_metrics=True)
