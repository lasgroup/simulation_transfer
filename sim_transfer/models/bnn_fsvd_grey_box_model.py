from sim_transfer.models.bnn_fsvgd import BNN_FSVGD
import jax.numpy as jnp
from typing import Optional, Dict
from collections import OrderedDict
import jax
from typing import NamedTuple
from sim_transfer.modules.distribution import AffineTransform
from sim_transfer.sims.simulators import FunctionSimulator


class BNN_FSVGD_GreyBox(BNN_FSVGD):

    def __init__(self, sim: FunctionSimulator, *args, **kwargs):
        super().__init__(domain=sim.domain, *args, **kwargs)
        self.sim = sim
        self.model_params = self.sim.init_params()

        self.params.update({'model_params': self.model_params})
        self._init_optim()

    def reinit(self, rng_key: Optional[jax.random.PRNGKey] = None):
        """ Reinitializes the model parameters and the optimizer state."""
        if rng_key is None:
            rng_key = self.rng_key
        key_model, key_rng = jax.random.split(rng_key)
        self._rng_key = key_rng  # reinitialize rng_key
        self.batched_model.reinit_params(key_model)  # reinitialize model parameters
        self.params['nn_params_stacked'] = self.batched_model.param_vectors_stacked
        self.params['model_params'] = self.model_params
        self._init_likelihood()  # reinitialize likelihood std
        self._init_optim()  # reinitialize optimizer

    def sim_model_step(self, x: jnp.array, params: NamedTuple):
        return self.sim.evaluate_sim(x, params)

    def predict_raw(self, x: jnp.ndarray, params: NamedTuple):
        f_raw = self.batched_model.forward_vec(x, params['nn_params_stacked'])
        unormalized_x = self._unnormalize_data(x)
        sim_model_prediction = self.sim_model_step(unormalized_x, params=params['model_params'])
        normalized_sim_model_predictions = self._normalize_y(sim_model_prediction)
        f_raw = jax.tree_util.tree_map(lambda y: y + normalized_sim_model_predictions, f_raw)
        return f_raw

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True):
        pred_dist_bnn = super().predict_dist(x, include_noise)
        sim_model_prediction = self.sim_model_step(x, params=self.params['model_params'])
        affine_transform_y = AffineTransform(shift=sim_model_prediction, scale=1.0)
        pred_dist = affine_transform_y(pred_dist_bnn)
        assert pred_dist.batch_shape == x.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        return pred_dist

    def predict_post_samples(self, x: jnp.ndarray) -> jnp.ndarray:
        sim_model_prediction = self.sim_model_step(x, params=self.params['model_params'])
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        y_pred = y_pred_raw * self._y_std + self._y_mean
        y_pred = jax.tree_util.tree_map(lambda y: y + sim_model_prediction, y_pred)
        assert y_pred.ndim == 3 and y_pred.shape[-2:] == (x.shape[0], self.output_size)
        return y_pred

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
        f_raw = self.predict_raw(x_stacked, params)

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
    obs_noise_std = 1e-3
    weight_decay = 1e-3
    x_train, y_train, x_test, y_test = sim.sample_datasets(rng_key=next(key_iter), num_samples_train=20000,
                                                           obs_noise_std=obs_noise_std, param_mode='typical')
    bnn = BNN_FSVGD_GreyBox(sim=sim, input_size=NUM_DIM_X, output_size=NUM_DIM_Y, rng_key=next(key_iter),
                            num_train_steps=20000,
                            bandwidth_svgd=1.0, likelihood_std=obs_noise_std, likelihood_exponent=1.0,
                            normalize_likelihood_std=True, learn_likelihood_std=False, weight_decay=weight_decay)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000, per_dim_metrics=True)
