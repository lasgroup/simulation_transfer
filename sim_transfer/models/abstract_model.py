import copy
import logging
import os
import pickle
import time
from typing import Optional, Tuple, Callable, Dict, List, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability.substrates.jax.distributions as tfd
import wandb
from jaxtyping import PyTree
from tensorflow_probability.substrates import jax as tfp

from sim_transfer.modules.distribution import AffineTransform
from sim_transfer.modules.metrics import _check_dist_and_sample_shapes, _get_mean_std_from_dist
from sim_transfer.modules.nn_modules import BatchedMLP
from sim_transfer.modules.util import RngKeyMixin, aggregate_stats


class AbstractRegressionModel(RngKeyMixin):

    def __init__(self, input_size: int, output_size: int, rng_key: jax.random.PRNGKey, normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None):
        super().__init__(rng_key)

        self.input_size = input_size
        self.output_size = output_size
        self.normalize_data = normalize_data
        self._need_to_compute_norm_stats = normalize_data and (normalization_stats is None)

        if (not normalize_data) or normalization_stats is None:
            # initialize normalization stats to neutral elements
            self._x_mean, self._x_std = jnp.zeros(input_size), jnp.ones(input_size)
            self._y_mean, self._y_std = jnp.zeros(output_size), jnp.ones(output_size)
        else:
            # check the provided normalization stats
            _n_stats = normalization_stats
            assert _n_stats['x_mean'].shape == _n_stats['x_std'].shape == (self.input_size,)
            assert _n_stats['y_mean'].shape == _n_stats['y_std'].shape == (self.output_size,)
            assert jnp.all(_n_stats['y_std'] > 0) and jnp.all(_n_stats['x_std'] > 0), 'stds need to be positive'
            # set the stats
            self._x_mean, self._x_std = _n_stats['x_mean'], _n_stats['x_std']
            self._y_mean, self._y_std = _n_stats['y_mean'], _n_stats['y_std']
        self.affine_transform_y = AffineTransform(shift=self._y_mean, scale=self._y_std)

        # disable some stupid tensorflow probability warnings
        self._add_checktypes_logging_filter()
        conf_values_cum = jnp.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.977, 0.99])
        quantiles_cum = tfp.distributions.Chi2(df=1).quantile(conf_values_cum)
        conf_values_bin = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
        quantiles_bin = tfp.distributions.Chi2(df=1).quantile(conf_values_bin)
        self.calibration_params = {
            'conf_values_cum': conf_values_cum,
            'quantiles_cum': quantiles_cum,
            'conf_values_bin': conf_values_bin,
            'quantiles_bin': quantiles_bin,
        }

        def step_jit(carry, ins):
            opt_state, params, key, num_train_points = carry[0], carry[1], carry[2], carry[3]
            x_batch, y_batch = ins[0], ins[1]
            key, train_key = jax.random.split(key)
            new_opt_state, new_params, stats = self._step_jit(
                opt_state=opt_state, params=params, x_batch=x_batch, y_batch=y_batch,
                key=train_key,
                num_train_points=num_train_points,
            )
            carry = [new_opt_state, new_params, key, num_train_points]
            return carry, stats

        self.step_jit = jax.jit(step_jit)

    def _compute_normalization_stats(self, x: jnp.ndarray, y: jnp.ndarray) -> None:
        # computes the empirical normalization stats and stores as private variables
        x, y = self._ensure_atleast_2d_float(x, y)
        self._x_mean = jnp.mean(x, axis=0)
        self._y_mean = jnp.mean(y, axis=0)
        self._x_std = jnp.std(x, axis=0)
        self._y_std = jnp.std(y, axis=0)

    def _normalize_data(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        # normalized the given data with the normalization stats
        if y is None:
            x = self._ensure_atleast_2d_float(x)
        else:
            x, y = self._ensure_atleast_2d_float(x, y)
        x_normalized = (x - self._x_mean[None, :]) / (self._x_std[None, :] + eps)
        assert x_normalized.shape == x.shape
        if y is None:
            return x_normalized
        else:
            y_normalized = self._normalize_y(y, eps=eps)
            assert y_normalized.shape == y.shape
            return x_normalized, y_normalized

    def _unnormalize_data(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        if y is None:
            x = self._ensure_atleast_2d_float(x)
        else:
            x, y = self._ensure_atleast_2d_float(x, y)
        x_unnorm = x * (self._x_std[None, :] + eps) + self._x_mean[None, :]
        assert x_unnorm.shape == x.shape
        if y is None:
            return x_unnorm
        else:
            y_unnorm = self._unnormalize_y(y, eps=eps)
            return x_unnorm, y_unnorm

    def _normalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y = self._ensure_atleast_2d_float(y)
        assert y.shape[-1] == self.output_size
        y_normalized = (y - self._y_mean) / (self._y_std + eps)
        assert y_normalized.shape == y.shape
        return y_normalized

    def _unnormalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y = self._ensure_atleast_2d_float(y)
        assert y.shape[-1] == self.output_size
        y_unnorm = y * (self._y_std + eps) + self._y_mean
        assert y_unnorm.shape == y.shape
        return y_unnorm

    @staticmethod
    def _ensure_atleast_2d_float(x: jnp.ndarray, y: Optional[jnp.ndarray] = None, dtype: jnp.dtype = jnp.float32):
        if x.ndim == 1:
            x = jnp.expand_dims(x, -1)
        if y is None:
            return jnp.array(x).astype(dtype=dtype)
        else:
            assert len(x) == len(y)
            if y.ndim == 1:
                y = jnp.expand_dims(y, -1)
            return jnp.array(x).astype(dtype=dtype), jnp.array(y).astype(dtype=dtype)

    def _to_pred_dist(self, y_pred_raw: jnp.ndarray, likelihood_std: jnp.ndarray, include_noise: bool = False):
        """ Forms the predictive distribution p(y|x, D) given the batched NNs raw outputs and the likelihood_std."""
        assert y_pred_raw.ndim == 3 and y_pred_raw.shape[-1] == self.output_size
        num_post_samples = y_pred_raw.shape[0]
        if include_noise:
            independent_normals = tfd.MultivariateNormalDiag(jnp.moveaxis(y_pred_raw, 0, 1), likelihood_std)
            mixture_distribution = tfd.Categorical(probs=jnp.ones(num_post_samples) / num_post_samples)
            pred_dist_raw = tfd.MixtureSameFamily(mixture_distribution, independent_normals)
        else:
            pred_dist_raw = tfd.MultivariateNormalDiag(jnp.mean(y_pred_raw, axis=0),
                                                       jnp.std(y_pred_raw, axis=0))
        pred_dist = self.affine_transform_y(pred_dist_raw)
        return pred_dist

    def _preprocess_train_data(self, x_train: jnp.ndarray, y_train: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Convert data to float32, ensure 2d shape and normalize if necessary"""
        if self.normalize_data:
            if self._need_to_compute_norm_stats:
                self._compute_normalization_stats(x_train, y_train)
            x_train, y_train = self._normalize_data(x_train, y_train)
            self.affine_transform_y = AffineTransform(shift=self._y_mean, scale=self._y_std)
        else:
            x_train, y_train = self._ensure_atleast_2d_float(x_train, y_train)
        return x_train, y_train

    def _create_data_loader(self, x_data: jnp.ndarray, y_data: jnp.ndarray,
                            batch_size: int = 64, shuffle: bool = True,
                            infinite: bool = True) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        if shuffle:
            seed = int(jax.random.randint(self.rng_key, (1,), 0, 10 ** 8))
            ds = ds.shuffle(batch_size * 4, seed=seed, reshuffle_each_iteration=True)
        if infinite:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = tfds.as_numpy(ds)
        return ds

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        raise NotImplementedError

    def _step_jit(self, opt_state: optax.OptState, params: PyTree, x_batch: jnp.array, y_batch: jnp.array,
                  key: jax.random.PRNGKey,
                  num_train_points: Union[float, int]) -> [optax.OptState, PyTree, Dict]:
        raise NotImplementedError

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = False) -> tfp.distributions.Distribution:
        return False

    def predict(self, x: jnp.ndarray, include_noise: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    def calibration_error_cum(self, dist: tfp.distributions.Distribution, y: jnp.ndarray) -> Union[jnp.ndarray, float]:
        """ Computes the calibration error of a distribution w.r.t. a sample based the differences in
            empirical cdf and expected cdf. """
        _check_dist_and_sample_shapes(dist, y)
        mean, std = _get_mean_std_from_dist(dist)
        res2 = ((mean - y) / std) ** 2
        conf_values = self.calibration_params['conf_values_cum']
        quantiles = self.calibration_params['quantiles_cum']
        emp_freqs = jnp.mean(res2[..., None] <= quantiles, axis=0)
        return jnp.mean(jnp.abs(emp_freqs - conf_values))

    def calibration_error_bin(self, dist: tfp.distributions.Distribution, y: jnp.ndarray) -> Union[jnp.ndarray, float]:
        """ Computes the calibration error of a distribution w.r.t. a sample based on 10 % bins of the probability mass """
        _check_dist_and_sample_shapes(dist, y)
        mean, std = _get_mean_std_from_dist(dist)
        res2 = ((mean - y) / std) ** 2
        conf_values = self.calibration_params['conf_values_bin']
        quantiles = self.calibration_params['quantiles_bin']
        n_bins = len(conf_values) - 1
        emp_freqs = jnp.mean((quantiles[:-1] <= res2[..., None]) & (res2[..., None] <= quantiles[1:]), axis=0)
        expected_freqs = conf_values[1:] - conf_values[:-1]
        return jnp.mean(jnp.sum(jnp.abs(emp_freqs - expected_freqs), axis=-1)) / (2 - 2 / n_bins)

    def eval(self, x: jnp.ndarray, y: np.ndarray, prefix: str = '', per_dim_metrics: bool = False) \
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
        pred_dist = self.predict_dist(x, include_noise=True)

        # compute standard evaluation metrics
        nll = - jnp.mean(pred_dist.log_prob(y))
        rmse = jnp.sqrt(jnp.mean(jnp.sum((pred_dist.mean - y) ** 2, axis=-1)))
        cal_err_bin = self.calibration_error_bin(pred_dist, y)
        cal_err_cum = self.calibration_error_cum(pred_dist, y)
        avg_std = jnp.mean(pred_dist.stddev)
        eval_stats = {'nll': nll, 'rmse': rmse, 'avg_std': avg_std,
                      'cal_err_cum': cal_err_cum, 'cal_err_bin': cal_err_bin}

        # compute per-dimension MAE
        if per_dim_metrics:
            mae_per_dim = jnp.mean(jnp.abs(pred_dist.mean - y), axis=0)
            rmse_per_dim = jnp.sqrt(jnp.mean((pred_dist.mean - y) ** 2, axis=0))
            eval_stats.update({f'per_dim_metrics/mae_{i}': mae_per_dim[i] for i in range(self.output_size)})
            eval_stats.update({f'per_dim_rmse/rmse_{i}': rmse_per_dim[i] for i in range(self.output_size)})

        # add prefix to stats names
        eval_stats = {f'{prefix}{name}': val for name, val in eval_stats.items()}
        # make sure that all stats are python native floats
        eval_stats = {name: float(val) for name, val in eval_stats.items()}
        return eval_stats

    def plot_1d(self,
                x_train: jnp.ndarray,
                y_train: jnp.ndarray,
                domain_l: Optional[float] = None,
                domain_u: Optional[float] = None,
                true_fun: Optional[Callable] = None,
                title: Optional[str] = '',
                show: bool = True,
                plot_posterior_samples: bool = False,
                log_to_wandb: bool = False,
                save_plot_dict: bool = False):
        assert self.input_size == 1, 'Can only plot if input_size = 1'

        plot_data = dict()

        # determine plotting domain
        x_min, x_max = jnp.min(x_train, axis=0), jnp.max(x_train, axis=0)
        width = x_max - x_min
        if domain_l is None:
            domain_l = x_min - 0.3 * width
        if domain_u is None:
            domain_u = x_max + 0.3 * width
        x_plot = jnp.linspace(domain_l, domain_u, 200).reshape((-1, 1))

        # make predictions
        pred_mean, pred_std = self.predict(x_plot)

        # draw the plot
        fig, ax = plt.subplots(nrows=1, ncols=self.output_size, figsize=(self.output_size * 4, 4))
        if self.output_size == 1:
            ax = [ax]
        for i in range(self.output_size):
            plot_data_dim = dict()
            plot_data_dim['Train points'] = (x_train.flatten(), y_train[:, i])
            ax[i].scatter(*plot_data_dim['Train points'], label='train points')
            if true_fun is not None:
                plot_data_dim['True function'] = (x_plot, true_fun(x_plot)[:, i])
                ax[i].plot(*plot_data_dim['True function'], label='true fun')

            plot_data_dim['Mean prediction'] = (x_plot.flatten(), pred_mean[:, i])
            plot_data_dim['Fill between std'] = (x_plot.flatten(), pred_mean[:, i] - pred_std[:, i],
                                                 pred_mean[:, i] + pred_std[:, i])
            ax[i].plot(*plot_data_dim['Mean prediction'], label='pred mean')
            ax[i].fill_between(*plot_data_dim['Fill between std'], alpha=0.3)

            if hasattr(self, 'predict_post_samples') and plot_posterior_samples:
                y_post_samples = self.predict_post_samples(x_plot)
                plot_data_dim['Sample predictions'] = (x_plot, y_post_samples[..., i])
                for y in y_post_samples:
                    ax[i].plot(x_plot, y[:, i], linewidth=0.2, color='green')

            if title is not None:
                ax[i].set_title(f'Output dimension {i}')
            ax[i].legend()
            plot_data[f'Dimension_{i}'] = plot_data_dim
        fig.suptitle(title)
        if log_to_wandb:
            wandb.log({title: wandb.Image(fig)})
            if save_plot_dict:
                directory = os.path.join(wandb.run.dir, 'data')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                data_path = os.path.join('data', 'plot_data.pkl')
                with open(os.path.join(wandb.run.dir, data_path), 'wb') as handle:
                    pickle.dump(data_path, handle)
                wandb.save(os.path.join(wandb.run.dir, data_path), wandb.run.dir)
        if show:
            fig.show()

    def _add_checktypes_logging_filter(self):
        logger = logging.getLogger()

        class CheckTypesFilter(logging.Filter):
            def filter(self, record):
                return "check_types" not in record.getMessage()

        logger.addFilter(CheckTypesFilter())


class BatchedNeuralNetworkModel(AbstractRegressionModel):

    def __init__(self,
                 *args,
                 data_batch_size: int = 16,
                 num_train_steps: int = 10000,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None,
                 num_batched_nns: int = 10,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._save_init_args(locals())
        self.data_batch_size = data_batch_size
        self.num_train_steps = num_train_steps
        self.num_batched_nns = num_batched_nns

        # setup batched mlp
        self.batched_model = BatchedMLP(input_size=self.input_size, output_size=self.output_size,
                                        num_batched_modules=num_batched_nns,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        hidden_activation=hidden_activation,
                                        last_activation=last_activation,
                                        rng_key=self.rng_key)

    def fit(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
            y_eval: Optional[jnp.ndarray] = None, num_steps: Optional[int] = None, log_period: int = 1000,
            log_to_wandb: bool = False, metrics_objective: str = 'train_nll_loss', keep_the_best: bool = False,
            per_dim_metrics: bool = False):
        # check whether eval data has been passed
        evaluate = x_eval is not None or y_eval is not None
        assert not evaluate or x_eval.shape[0] == y_eval.shape[0]

        # prepare data and data loader
        x_train, y_train = self._preprocess_train_data(x_train, y_train)
        batch_size = min(self.data_batch_size, x_train.shape[0])
        train_loader = self._create_data_loader(x_train, y_train, batch_size=batch_size)
        num_train_points = x_train.shape[0]

        # initialize attributes to keep track of during training
        num_steps = self.num_train_steps if num_steps is None else num_steps
        samples_cum_period = 0.0
        stats_list = []
        t_start_period = time.time()

        # do callback before training loop starts in case the specific method wants to do something for
        # which it needs the normalization stats of the data
        self._before_training_loop_callback()

        best_objective = jnp.array(jnp.inf)
        best_params = None

        # training loop
        for step, (x_batch, y_batch) in enumerate(train_loader, 1):
            samples_cum_period += x_batch.shape[0]

            # perform the train step
            stats = self.step(x_batch, y_batch, num_train_points)
            stats_list.append(stats)

            if step % log_period == 0 or step == 1:
                duration_sec = time.time() - t_start_period
                duration_per_sample_ms = duration_sec / samples_cum_period * 1000
                stats_agg = aggregate_stats(stats_list)
                if evaluate:
                    eval_stats = self.eval(x_eval, y_eval, prefix='eval_', per_dim_metrics=per_dim_metrics)
                    if keep_the_best:
                        if metrics_objective in eval_stats.keys():
                            if eval_stats[metrics_objective] < best_objective:
                                best_objective = eval_stats[metrics_objective]
                                best_params = copy.deepcopy(self.params)
                        else:
                            if stats[metrics_objective] < best_objective:
                                best_objective = stats[metrics_objective]
                                best_params = copy.deepcopy(self.params)
                    stats_agg.update(eval_stats)
                if log_to_wandb:
                    log_dict = {f'regression_model_training/{n}': float(v) for n, v in stats_agg.items()}
                    wandb.log(log_dict | {'x_axis/bnn_step': step})
                stats_msg = ' | '.join([f'{n}: {v:.4f}' for n, v in stats_agg.items()])
                msg = (f'Step {step}/{num_steps} | {stats_msg} | Duration {duration_sec:.2f} sec | '
                       f'Time per sample {duration_per_sample_ms:.2f} ms')
                print(msg)

                # reset the attributes we keep track of
                stats_list = []
                samples_cum_period = 0
                t_start_period = time.time()

            if step >= num_steps:
                break

        if keep_the_best and best_params is not None:
            self.params = best_params
            print(f'Keeping the best model with {metrics_objective}={best_objective:.4f}')
            if log_to_wandb:
                wandb.log({metrics_objective: best_objective})

    def fit_with_scan(self,
                      x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
                      y_eval: Optional[jnp.ndarray] = None, num_steps: Optional[int] = None, log_period: int = 1000,
                      log_to_wandb: bool = False, metrics_objective: str = 'train_nll_loss',
                      keep_the_best: bool = False,
                      per_dim_metrics: bool = False
                      ):
        evaluate = x_eval is not None or y_eval is not None
        assert not evaluate or x_eval.shape[0] == y_eval.shape[0]

        # prepare data and data loader
        x_train, y_train = self._preprocess_train_data(x_train, y_train)
        batch_size = min(self.data_batch_size, x_train.shape[0])
        num_train_points = x_train.shape[0]

        # initialize attributes to keep track of during training
        num_steps = self.num_train_steps if num_steps is None else num_steps
        t_start_period = time.time()

        # do callback before training loop starts in case the specific method wants to do something for
        # which it needs the normalization stats of the data
        self._before_training_loop_callback()

        best_objective = jnp.array(jnp.inf)
        best_params = None
        train_steps = min(num_steps, log_period)
        log_loops = num_steps // train_steps

        for i in range(log_loops):
            batch_index = jax.random.randint(minval=0, maxval=num_train_points,
                                             key=self.rng_key, shape=(train_steps, batch_size))
            x_data, y_data = jax.vmap(lambda idx: (x_train[idx], y_train[idx]))(batch_index)
            carry, outs = jax.lax.scan(self.step_jit, [self.opt_state, self.params, self.rng_key, num_train_points],
                                       xs=[x_data, y_data],
                                       length=train_steps)
            self.opt_state, self.params = carry[0], carry[1]
            stats = jax.tree_util.tree_map(lambda x: x[-1], outs)
            step = (i + 1) * train_steps
            samples_cum_period = (i + 1) * train_steps * x_train.shape[0]
            duration_sec = time.time() - t_start_period
            duration_per_sample_ms = duration_sec / samples_cum_period * 1000
            stats_agg = jax.tree_util.tree_map(jnp.mean, outs)
            if step % log_period == 0:
                if evaluate:
                    eval_stats = self.eval(x_eval, y_eval, prefix='eval_', per_dim_metrics=per_dim_metrics)
                    if keep_the_best:
                        if metrics_objective in eval_stats.keys():
                            if eval_stats[metrics_objective] < best_objective:
                                best_objective = eval_stats[metrics_objective]
                                best_params = copy.deepcopy(self.params)
                        else:
                            if stats[metrics_objective] < best_objective:
                                best_objective = stats[metrics_objective]
                                best_params = copy.deepcopy(self.params)
                    stats_agg.update(eval_stats)
                if log_to_wandb:
                    log_dict = {f'regression_model_training/{n}': float(v) for n, v in stats_agg.items()}
                    wandb.log(log_dict | {'x_axis/bnn_step': step})
                stats_msg = ' | '.join([f'{n}: {v:.4f}' for n, v in stats_agg.items()])
                msg = (f'Step {step}/{num_steps} | {stats_msg} | Duration {duration_sec:.2f} sec | '
                       f'Time per sample {duration_per_sample_ms:.2f} ms')
                print(msg)

                # reset the attributes we keep track of
                t_start_period = time.time()
        if keep_the_best and best_params is not None:
            self.params = best_params
            print(f'Keeping the best model with {metrics_objective}={best_objective:.4f}')
            if log_to_wandb:
                wandb.log({metrics_objective: best_objective})

    def predict(self, x: jnp.ndarray, include_noise: bool = False, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Returns the mean and std of the predictive distribution.

        Args:
            x (jnp.ndarray): points in which to make predictions
            include_noise (bool): whether to include alleatoric uncertainty in the predictive std. If False,
                                  only return the epistemic std.

        Returns: predictive mean and std
        """
        pred_dist = self.predict_dist(x=x, include_noise=include_noise, **kwargs)
        return pred_dist.mean, pred_dist.stddev

    def _before_training_loop_callback(self) -> None:
        """
        Called right before the training loop starts.
        """
        pass

    def _construct_nn_param_prior(self, weight_prior_std: float, bias_prior_std: float) -> tfd.MultivariateNormalDiag:
        return self.batched_model.params_prior(weight_prior_std=weight_prior_std, bias_prior_std=bias_prior_std)

    @staticmethod
    def _activation_fn_to_str(activation_fn: Callable) -> str:
        if activation_fn is None:
            return 'None'
        elif 'leaky_relu' in activation_fn.__str__():
            return 'leaky_relu'
        elif 'relu' in activation_fn.__str__():
            return 'relu'
        elif 'tanh' in activation_fn.__str__():
            return 'tanh'
        elif 'sigmoid' in activation_fn.__str__():
            return 'sigmoid'
        elif 'softplus' in activation_fn.__str__():
            return 'softplus'
        elif 'elu' in activation_fn.__str__():
            return 'elu'
        elif 'selu' in activation_fn.__str__():
            return 'selu'

    @staticmethod
    def _str_to_activation_fn(activation_fn_str: str) -> Callable:
        if activation_fn_str == 'None':
            return None
        elif activation_fn_str == 'leaky_relu':
            return jax.nn.leaky_relu
        elif activation_fn_str == 'relu':
            return jax.nn.relu
        elif activation_fn_str == 'tanh':
            return jnp.tanh
        elif activation_fn_str == 'sigmoid':
            return jax.nn.sigmoid
        elif activation_fn_str == 'softplus':
            return jax.nn.softplus
        elif activation_fn_str == 'elu':
            return jax.nn.elu
        elif activation_fn_str == 'selu':
            return jax.nn.selu

    def _save_init_args(self, local_var_dict: Dict) -> None:
        self._init_arguments = {k: v for k, v in local_var_dict.items() if not (k.startswith('__') or k == 'self')}

    def _set_state(self, state_dict: Dict) -> None:
        for var_name, val in state_dict.items():
            setattr(self, var_name, val)

    def _get_state(self) -> Dict:
        raise NotImplementedError

    def __getstate__(self):
        init_args = self._init_arguments.copy()
        init_args['hidden_activation'] = self._activation_fn_to_str(init_args['hidden_activation'])
        init_args['last_activation'] = self._activation_fn_to_str(init_args['last_activation'])
        return {'init_args': init_args, 'state': self._get_state()}

    def __setstate__(self, state):
        init_args = state['init_args']
        init_args['hidden_activation'] = self._str_to_activation_fn(init_args['hidden_activation'])
        init_args['last_activation'] = self._str_to_activation_fn(init_args['last_activation'])
        self.__init__(**init_args)
        self._set_state(state['state'])
