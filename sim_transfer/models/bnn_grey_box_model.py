from sim_transfer.models.abstract_model import AbstractRegressionModel
from sim_transfer.models.bnn import AbstractParticleBNN
from sim_transfer.models.bnn_fsvgd import BNN_FSVGD
import jax.numpy as jnp
from typing import Optional, Dict, Union, Tuple
from collections import OrderedDict
import jax
from typing import NamedTuple
from sim_transfer.sims.simulators import FunctionSimulator
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from functools import partial
import time
import copy
from sim_transfer.modules.util import aggregate_stats
import wandb
import numpy as np
from tensorflow_probability.substrates import jax as tfp


class BNNGreyBox(AbstractRegressionModel):

    def __init__(self,
                 base_bnn: AbstractParticleBNN,
                 sim: FunctionSimulator,
                 lr_sim: float = None,
                 weight_decay_sim: float = 0.0,
                 num_sim_model_train_steps: int = 10_000,
                 use_base_bnn: bool = True):
        self.base_bnn = base_bnn
        super().__init__(
            input_size=self.base_bnn.input_size,
            output_size=self.base_bnn.output_size,
            rng_key=self.base_bnn.rng_key,
            normalize_data=False,
            normalization_stats=None,
        )
        del self._x_mean, self._x_std, self._y_mean, self._y_std, self.affine_transform_y

        def step_jit_sim(carry, ins):
            opt_state_sim, params_sim, num_train_points = carry[0], carry[1], carry[2]
            x_batch, y_batch = ins[0], ins[1]
            new_opt_state_sim, new_params_sim, stats = self._step_grey_box_jit(
                opt_state_sim=opt_state_sim, params_sim=params_sim, x_batch=x_batch,
                y_batch=y_batch, num_train_points=num_train_points)
            carry = [new_opt_state_sim, new_params_sim, num_train_points]
            return carry, stats

        self.step_jit_sim = jax.jit(step_jit_sim)
        self.normalize_data = self.base_bnn.normalize_data
        self._need_to_compute_norm_stats = self.base_bnn._need_to_compute_norm_stats
        self.sim = sim
        param_key = self._next_rng_key()
        self.num_sim_model_train_steps = num_sim_model_train_steps
        sim_params, train_params = self.sim.sample_params(param_key)
        likelihood_std_raw = -1. * jnp.ones(self.output_size)
        self.init_likelihood_std_raw = likelihood_std_raw
        self.params_sim = {'sim_params': sim_params, 'likelihood_std_raw': likelihood_std_raw}
        self.init_sim_params = sim_params
        self.train_params = train_params
        self.optim_sim = None
        self._x_mean_sim, self._x_std_sim = jnp.copy(self.base_bnn._x_mean), jnp.copy(self.base_bnn._x_std)
        self._y_mean_sim, self._y_std_sim = jnp.copy(self.base_bnn._y_mean), jnp.copy(self.base_bnn._y_std)
        self.use_base_bnn = use_base_bnn
        if lr_sim:
            self.lr_sim = lr_sim
        else:
            self.lr_sim = self.base_bnn.lr
        if weight_decay_sim is not None:
            self.weight_decay_sim = weight_decay_sim
        else:
            self.weight_decay_sim = self.base_bnn.weight_decay
        self._init_optim()

    def _init_sim_optim(self):
        """ Initializes the optimizer and the optimizer state.
        Sets the attributes self.optim and self.opt_state. """
        if self.weight_decay_sim > 0:
            self.optim_sim = optax.adamw(learning_rate=self.lr_sim, weight_decay=self.weight_decay_sim)
        else:
            self.optim_sim = optax.adam(learning_rate=self.lr_sim)

        self.opt_state_sim = self.optim_sim.init(self.params_sim)

    def _init_optim(self):
        self.base_bnn._init_optim()
        self._init_sim_optim()

    def reinit(self, rng_key: Optional[jax.random.PRNGKey] = None):
        """ Reinitializes the model parameters and the optimizer state."""
        if rng_key is None:
            key_rng = self._next_rng_key()
            key_model = self._next_rng_key()
            param_key = self._next_rng_key()
        else:
            key_model, key_rng, param_key = jax.random.split(rng_key, 3)
        self.base_bnn.reinit(key_model)
        self._rng_key = key_rng  # reinitialize rng_key
        sim_params, train_params = self.sim.sample_params(param_key)
        likelihood_std_raw = -1. * jnp.ones(self.output_size)
        self.init_sim_params = sim_params
        self.train_params = train_params
        self.params_sim = {'sim_params': sim_params, 'likelihood_std_raw': likelihood_std_raw}
        self._init_optim()  # reinitialize optimizer

    @property
    def params(self):
        return self.base_bnn.params

    @property
    def batched_model(self):
        return self.base_bnn.batched_model

    @property
    def likelihood_std(self):
        if self.use_base_bnn:
            return self.base_bnn.likelihood_std
        else:
            return self.sim_likelihood_std

    @property
    def likelihood_std_unnormalized(self):
        if self.use_base_bnn:
            return self.base_bnn.likelihood_std_unnormalized
        else:
            return self.sim_likelihood_std_unnormalized

    @property
    def sim_likelihood_std(self):
        if self.learn_likelihood_std:
            likelihood_std = jax.nn.softplus(self.params_sim['likelihood_std_raw'])
        else:
            likelihood_std = jax.nn.softplus(self.init_likelihood_std_raw)
        return likelihood_std

    @property
    def sim_likelihood_std_unnormalized(self):
        likelihood_std = self.sim_likelihood_std
        assert hasattr(self, '_y_std_sim') and self._y_std_sim is not None and self.normalize_data, \
            'normalize_likelihood_std requires normalization'
        assert self._y_std_sim.shape == (self.output_size,)
        likelihood_std = likelihood_std * self._y_std_sim
        return likelihood_std

    @property
    def learn_likelihood_std(self):
        return self.base_bnn.learn_likelihood_std

    @property
    def data_batch_size(self):
        return self.base_bnn.data_batch_size

    @property
    def likelihood_exponent(self):
        return self.base_bnn.likelihood_exponent

    def _normalize_data(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        # normalized the given data with the normalization stats
        return self.base_bnn._normalize_data(x, y, eps)

    def _unnormalize_data(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        return self.base_bnn._unnormalize_data(x, y, eps)

    def _normalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        return self.base_bnn._normalize_y(y, eps)

    def _unnormalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        return self.base_bnn._unnormalize_y(y, eps)

    def _preprocess_train_data_sim(self, x_train: jnp.ndarray, y_train: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Convert data to float32, ensure 2d shape and normalize if necessary"""
        if self.normalize_data:
            if self._need_to_compute_norm_stats:
                self._compute_normalization_stats_sim(x_train, y_train)
            x_train, y_train = self._normalize_data_sim(x_train, y_train)
        else:
            x_train, y_train = self._ensure_atleast_2d_float(x_train, y_train)
        return x_train, y_train

    def _compute_normalization_stats_sim(self, x: jnp.ndarray, y: jnp.ndarray) -> None:
        # computes the empirical normalization stats and stores as private variables
        x, y = self._ensure_atleast_2d_float(x, y)
        self._x_mean_sim = jnp.mean(x, axis=0)
        self._y_mean_sim = jnp.mean(y, axis=0)
        self._x_std_sim = jnp.std(x, axis=0)
        self._y_std_sim = jnp.std(y, axis=0)

    def _normalize_data_sim(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        # normalized the given data with the normalization stats
        if y is None:
            x = self._ensure_atleast_2d_float(x)
        else:
            x, y = self._ensure_atleast_2d_float(x, y)
        x_normalized = (x - self._x_mean_sim[None, :]) / (self._x_std_sim[None, :] + eps)
        assert x_normalized.shape == x.shape
        if y is None:
            return x_normalized
        else:
            y_normalized = self._normalize_y_sim(y, eps=eps)
            assert y_normalized.shape == y.shape
            return x_normalized, y_normalized

    def _unnormalize_data_sim(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        if y is None:
            x = self._ensure_atleast_2d_float(x)
        else:
            x, y = self._ensure_atleast_2d_float(x, y)
        x_unnorm = x * (self._x_std_sim[None, :] + eps) + self._x_mean_sim[None, :]
        assert x_unnorm.shape == x.shape
        if y is None:
            return x_unnorm
        else:
            y_unnorm = self._unnormalize_y_sim(y, eps=eps)
            return x_unnorm, y_unnorm

    def _normalize_y_sim(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y = self._ensure_atleast_2d_float(y)
        assert y.shape[-1] == self.output_size
        y_normalized = (y - self._y_mean_sim) / (self._y_std_sim + eps)
        assert y_normalized.shape == y.shape
        return y_normalized

    def _unnormalize_y_sim(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y = self._ensure_atleast_2d_float(y)
        assert y.shape[-1] == self.output_size
        y_unnorm = y * (self._y_std_sim + eps) + self._y_mean_sim
        assert y_unnorm.shape == y.shape
        return y_unnorm

    @partial(jax.jit, static_argnums=0)
    def sim_model_step(self, x: jnp.array, params_sim: NamedTuple):
        """Take unnormalized inputs and return unnormalized output from the sim."""
        # parameters that are trainable, have 1 in self.train_params if train_params is 0 then initial param is taken.
        params_sim = jax.tree_util.tree_map(lambda v, w, z:
                                            v * w + (1 - w) * z, params_sim, self.train_params, self.init_sim_params)
        y = self.sim.evaluate_sim(x, params_sim)
        return y

    def _sim_loss(self, params_sim: Dict, x_batch: jnp.array, y_batch: jnp.array,
                  num_train_points: Union[float, int]):
        x = self._unnormalize_data_sim(x_batch)
        sim_model_prediction = self.sim_model_step(x, params_sim['sim_params'])

        normalized_sim_model_prediction = self._normalize_y_sim(sim_model_prediction)
        assert normalized_sim_model_prediction.shape == sim_model_prediction.shape
        # get likelihood std
        likelihood_std = jax.nn.softplus(params_sim['likelihood_std_raw']) if self.learn_likelihood_std \
            else jax.nn.softplus(self.init_likelihood_std_raw)

        ll = tfd.MultivariateNormalDiag(sim_model_prediction, likelihood_std).log_prob(y_batch)
        nll = - num_train_points * self.likelihood_exponent * jnp.mean(ll, axis=0)
        return nll

    def _sim_step(self, opt_state_sim: optax.OptState, params_sim: Dict,
                  x_batch: jnp.array, y_batch: jnp.array, num_train_points: Union[float, int]):
        loss, grad = jax.value_and_grad(
            self._sim_loss)(
            params_sim,
            x_batch, y_batch,
            num_train_points)
        updates, new_opt_state_sim = self.optim_sim.update(grad, opt_state_sim, params_sim)
        new_params_sim = optax.apply_updates(params_sim, updates)
        stats = OrderedDict(sim_params_nll=loss)
        return new_opt_state_sim, new_params_sim, stats

    def _step_grey_box(self, opt_state_sim: optax.OptState, params_sim: Dict,
                       x_batch: jnp.array, y_batch: jnp.array,
                       num_train_points: Union[float, int]):
        new_opt_state_sim, new_params_sim, stats = self._sim_step(
            opt_state_sim, params_sim, x_batch, y_batch, num_train_points
        )
        return new_opt_state_sim, new_params_sim, stats

    @partial(jax.jit, static_argnums=(0,))
    def _step_grey_box_jit(self, *args, **kwargs):
        return self._step_grey_box(*args, **kwargs)

    def step_grey_box(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]):
        self.opt_state_sim, self.params_sim, stats = self._step_grey_box_jit(
            opt_state_sim=self.opt_state_sim, params_sim=self.params_sim, x_batch=x_batch,
            y_batch=y_batch, num_train_points=num_train_points)
        return stats

    def predict_post_samples(self, x: jnp.ndarray) -> jnp.ndarray:
        sim_model_prediction = self.sim_model_step(x, self.params_sim['sim_params'])
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        y_pred = jax.vmap(lambda y: int(self.use_base_bnn) * self._unnormalize_y(y) + sim_model_prediction,
                          in_axes=0)(y_pred_raw)
        assert y_pred.ndim == 3 and y_pred.shape[-2:] == (x.shape[0], self.output_size)
        return y_pred

    def _to_pred_dist(self, y_pred_raw: jnp.ndarray, likelihood_std: jnp.ndarray, include_noise: bool = False):
        """ Forms the predictive distribution p(y|x, D) given the models unnormalized outputs and the likelihood_std."""
        assert y_pred_raw.ndim == 3 and y_pred_raw.shape[-1] == self.output_size
        num_post_samples = y_pred_raw.shape[0]
        if include_noise:
            independent_normals = tfd.MultivariateNormalDiag(jnp.moveaxis(y_pred_raw, 0, 1), likelihood_std)
            mixture_distribution = tfd.Categorical(probs=jnp.ones(num_post_samples) / num_post_samples)
            pred_dist = tfd.MixtureSameFamily(mixture_distribution, independent_normals)
        else:
            pred_dist = tfd.MultivariateNormalDiag(jnp.mean(y_pred_raw, axis=0),
                                                   jnp.std(y_pred_raw, axis=0))
        return pred_dist

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True) -> tfp.distributions.Distribution:
        self.batched_model.param_vectors_stacked = self.params['nn_params_stacked']
        y_pred = self.predict_post_samples(x)
        pred_dist = self._to_pred_dist(y_pred, likelihood_std=self.likelihood_std_unnormalized,
                                       include_noise=include_noise)
        assert pred_dist.batch_shape == x.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        if callable(pred_dist.mean):
            mean, stddev, var = pred_dist.mean(), pred_dist.stddev(), pred_dist.variance()
            pred_dist.mean = mean
            pred_dist.stddev = stddev
            pred_dist.variance = var
        return pred_dist

    def predict(self, x: jnp.ndarray, include_noise: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pred_dist = self.predict_dist(x=x, include_noise=include_noise)
        return pred_dist.mean, pred_dist.stddev

    def eval_sim(self, x: jnp.ndarray, y: np.ndarray, prefix: str = '', per_dim_metrics: bool = False) \
            -> Dict[str, jnp.ndarray]:
        """ Evaluates the model on the given data and returns a dictionary of evaluation metrics.
        Args:
            x: inputs
            y: targets
            prefix: (str) prefix for the returned dictionary keys
            per_dim_metrics: (bool) whether to also return the per-dimension MAE

        Returns: dict of evaluation metrics
        """
        # make predictions
        x, y = self._ensure_atleast_2d_float(x, y)

        if self.use_base_bnn:
            pred_y = self.sim_model_step(x, self.params_sim['sim_params'])
            rmse = jnp.sqrt(jnp.mean(jnp.sum((pred_y - y) ** 2, axis=-1)))
            eval_stats = {'rmse': rmse}
        else:
            pred_dist = self.predict_dist(x, include_noise=True)
            nll = - jnp.mean(pred_dist.log_prob(y))
            rmse = jnp.sqrt(jnp.mean(jnp.sum((pred_dist.mean - y) ** 2, axis=-1)))
            avg_likelihood_std = jnp.mean(self.likelihood_std_unnormalized)
            eval_stats = {'rmse': rmse, 'nll': nll, 'likelihood_std': avg_likelihood_std}

            print('likelihood_stds', self.likelihood_std_unnormalized)

        # compute per-dimension MAE
        if per_dim_metrics:
            mae_per_dim = jnp.mean(jnp.abs(pred_y - y), axis=0)
            rmse_per_dim = jnp.sqrt(jnp.mean((pred_y - y) ** 2, axis=0))
            eval_stats.update({f'per_dim_metrics/mae_{i}': mae_per_dim[i] for i in range(self.output_size)})
            eval_stats.update({f'per_dim_rmse/rmse_{i}': rmse_per_dim[i] for i in range(self.output_size)})

        # add prefix to stats names
        eval_stats = {f'{prefix}{name}': val for name, val in eval_stats.items()}
        # make sure that all stats are python native floats
        eval_stats = {name: float(val) for name, val in eval_stats.items()}
        return eval_stats

    def fit_sim_prior(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
                      y_eval: Optional[jnp.ndarray] = None,
                      num_sim_model_train_steps: Optional[int] = None, log_period: int = 1000,
                      log_to_wandb: bool = False, metrics_objective: str = 'train_sim_nll_loss',
                      keep_the_best: bool = False,
                      per_dim_metrics: bool = False):

        evaluate = x_eval is not None or y_eval is not None
        assert not evaluate or x_eval.shape[0] == y_eval.shape[0]

        # prepare data and data loader
        x_train, y_train = self._preprocess_train_data_sim(x_train, y_train)
        batch_size = min(self.data_batch_size, x_train.shape[0])
        train_loader = self._create_data_loader(x_train, y_train, batch_size=batch_size)
        num_train_points = x_train.shape[0]

        num_sim_model_train_steps = self.num_sim_model_train_steps if num_sim_model_train_steps is None else \
            num_sim_model_train_steps
        samples_cum_period = 0.0
        stats_list = []
        t_start_period = time.time()

        best_objective = jnp.array(jnp.inf)
        best_params = None

        # training loop
        for step, (x_batch, y_batch) in enumerate(train_loader, 1):
            samples_cum_period += x_batch.shape[0]

            # perform the train step

            stats = self.step_grey_box(x_batch, y_batch, num_train_points)
            stats_list.append(stats)

            if step % log_period == 0 or step == 1:
                duration_sec = time.time() - t_start_period
                duration_per_sample_ms = duration_sec / samples_cum_period * 1000
                stats_agg = aggregate_stats(stats_list)
                if evaluate:
                    eval_stats = self.eval_sim(x_eval, y_eval, prefix='eval_', per_dim_metrics=per_dim_metrics)
                    if keep_the_best:
                        if metrics_objective in eval_stats.keys():
                            if eval_stats[metrics_objective] < best_objective:
                                best_objective = eval_stats[metrics_objective]
                                best_params = copy.deepcopy(self.params_sim)
                        else:
                            if stats['sim_params_nll'] < best_objective:
                                best_objective = stats['sim_params_nll']
                                best_params = copy.deepcopy(self.params_sim)
                    stats_agg.update(eval_stats)
                if log_to_wandb:
                    log_dict = {f'regression_model_training/{n}': float(v) for n, v in stats_agg.items()}
                    wandb.log(log_dict | {'x_axis/bnn_step': step})
                stats_msg = ' | '.join([f'{n}: {v:.4f}' for n, v in stats_agg.items()])
                msg = (f'Step {step}/{num_sim_model_train_steps} | {stats_msg} | Duration {duration_sec:.2f} sec | '
                       f'Time per sample {duration_per_sample_ms:.2f} ms')
                print(msg)

                # reset the attributes we keep track of
                stats_list = []
                samples_cum_period = 0
                t_start_period = time.time()

            if step >= num_sim_model_train_steps:
                break

        if keep_the_best and best_params is not None:
            self.params_sim = best_params
            print(f'Keeping the best model with {metrics_objective}={best_objective:.4f}')
            if log_to_wandb:
                wandb.log({metrics_objective: best_objective})

    def fit_sim_prior_with_scan(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
                                y_eval: Optional[jnp.ndarray] = None,
                                num_sim_model_train_steps: Optional[int] = None, log_period: int = 1000,
                                log_to_wandb: bool = False, metrics_objective: str = 'train_sim_nll_loss',
                                keep_the_best: bool = False,
                                per_dim_metrics: bool = False):
        evaluate = x_eval is not None or y_eval is not None
        assert not evaluate or x_eval.shape[0] == y_eval.shape[0]

        x_train, y_train = self._preprocess_train_data_sim(x_train, y_train)
        batch_size = min(self.data_batch_size, x_train.shape[0])
        num_train_points = x_train.shape[0]

        num_sim_model_train_steps = self.num_sim_model_train_steps if num_sim_model_train_steps is None else \
            num_sim_model_train_steps

        best_objective = jnp.array(jnp.inf)
        best_params = None
        train_steps = min(num_sim_model_train_steps, log_period)
        log_loops = num_sim_model_train_steps // train_steps
        t_start_period = time.time()

        for i in range(log_loops):
            batch_index = jax.random.randint(minval=0, maxval=num_train_points,
                                             key=self.rng_key, shape=(train_steps, batch_size))
            x_data, y_data = jax.vmap(lambda idx: (x_train[idx], y_train[idx]))(batch_index)
            carry, outs = jax.lax.scan(self.step_jit_sim, [self.opt_state_sim, self.params_sim, num_train_points],
                                       xs=[x_data, y_data],
                                       length=train_steps)
            self.opt_state_sim, self.params_sim = carry[0], carry[1]
            stats = jax.tree_util.tree_map(lambda x: x[-1], outs)
            step = (i + 1) * train_steps
            samples_cum_period = (i + 1) * train_steps * x_train.shape[0]
            duration_sec = time.time() - t_start_period
            duration_per_sample_ms = duration_sec / samples_cum_period * 1000
            stats_agg = jax.tree_util.tree_map(jnp.mean, outs)
            if step % log_period == 0:
                if evaluate:
                    eval_stats = self.eval_sim(x_eval, y_eval, prefix='eval_', per_dim_metrics=per_dim_metrics)
                    if keep_the_best:
                        if metrics_objective in eval_stats.keys():
                            if eval_stats[metrics_objective] < best_objective:
                                best_objective = eval_stats[metrics_objective]
                                best_params = copy.deepcopy(self.params_sim)
                        else:
                            if stats['sim_params_nll'] < best_objective:
                                best_objective = stats['sim_params_nll']
                                best_params = copy.deepcopy(self.params_sim)
                    stats_agg.update(eval_stats)
                if log_to_wandb:
                    log_dict = {f'regression_model_training/{n}': float(v) for n, v in stats_agg.items()}
                    wandb.log(log_dict | {'x_axis/bnn_step': step})
                stats_msg = ' | '.join([f'{n}: {v:.4f}' for n, v in stats_agg.items()])
                msg = (f'Step {step}/{num_sim_model_train_steps} | {stats_msg} | Duration {duration_sec:.2f} sec | '
                       f'Time per sample {duration_per_sample_ms:.2f} ms')
                print(msg)

                # reset the attributes we keep track of
                t_start_period = time.time()
        if keep_the_best and best_params is not None:
            self.params_sim = best_params
            print(f'Keeping the best model with {metrics_objective}={best_objective:.4f}')
            if log_to_wandb:
                wandb.log({metrics_objective: best_objective})

    def fit_with_scan(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
                      y_eval: Optional[jnp.ndarray] = None, num_steps: Optional[int] = None,
                      num_sim_model_train_steps: Optional[int] = None, log_period: int = 1000,
                      log_to_wandb: bool = False, metrics_objective: str = 'train_nll_loss',
                      keep_the_best: bool = False,
                      per_dim_metrics: bool = False):

        self.fit_sim_prior_with_scan(x_train, y_train, x_eval, y_eval, num_sim_model_train_steps, log_period,
                                     log_to_wandb, metrics_objective, keep_the_best, per_dim_metrics)
        y_train = y_train - self.sim_model_step(x_train, self.params_sim['sim_params'])
        y_eval = y_eval - self.sim_model_step(x_eval, self.params_sim['sim_params'])
        if self.use_base_bnn:
            self.base_bnn.fit_with_scan(
                x_train, y_train, x_eval, y_eval, num_steps, log_period, log_to_wandb,
                metrics_objective,
                keep_the_best, per_dim_metrics
            )

    def fit(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
            y_eval: Optional[jnp.ndarray] = None, num_steps: Optional[int] = None,
            num_sim_model_train_steps: Optional[int] = None, log_period: int = 1000,
            log_to_wandb: bool = False, metrics_objective: str = 'train_nll_loss', keep_the_best: bool = False,
            per_dim_metrics: bool = False):

        self.fit_sim_prior(x_train, y_train, x_eval, y_eval, num_sim_model_train_steps, log_period,
                           log_to_wandb, metrics_objective, keep_the_best, per_dim_metrics)
        y_train = y_train - self.sim_model_step(x_train, self.params_sim['sim_params'])
        y_eval = y_eval - self.sim_model_step(x_eval, self.params_sim['sim_params'])
        if self.use_base_bnn:
            self.base_bnn.fit(
                x_train, y_train, x_eval, y_eval, num_steps, log_period, log_to_wandb,
                metrics_objective,
                keep_the_best, per_dim_metrics
            )


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
    weight_decay = 1e-5
    x_train, y_train, x_test, y_test = sim.sample_datasets(rng_key=next(key_iter), num_samples_train=20000,
                                                           obs_noise_std=obs_noise_std, param_mode='typical')
    base_bnn = BNN_FSVGD(
        input_size=NUM_DIM_X, output_size=NUM_DIM_Y, rng_key=next(key_iter),
        num_train_steps=20000,
        bandwidth_svgd=1.0, likelihood_std=obs_noise_std, likelihood_exponent=1.0,
        normalize_likelihood_std=True, learn_likelihood_std=True, weight_decay=weight_decay, domain=sim.domain,
    )
    bnn = BNNGreyBox(base_bnn=base_bnn, sim=sim, use_base_bnn=True, lr_sim=3e-4)
    for i in range(10):
        bnn.fit_with_scan(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000, per_dim_metrics=True,
                          num_sim_model_train_steps=2000)
        y_pred, _ = bnn.predict(x_test)
        loss = jnp.sqrt(jnp.square(y_pred - y_test).sum(axis=-1)).mean(0)
        print('loss: ', loss)
