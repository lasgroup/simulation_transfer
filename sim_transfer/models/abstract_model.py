import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Callable, Dict


import time
import matplotlib.pyplot as plt
import numpy as np
import logging

from sim_transfer.modules.util import RngKeyMixin, aggregate_stats
from sim_transfer.modules.distribution import AffineTransform

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd


class AbstactRegressionModel(RngKeyMixin):

    def __init__(self, input_size: int, output_size: int, rng_key: jax.random.PRNGKey, normalize_data: bool = True):
        super().__init__(rng_key)

        self.input_size = input_size
        self.output_size = output_size
        self.normalize_data = normalize_data

        # initialize normalization stats to neutral elements
        self._x_mean, self._x_std = jnp.zeros(input_size), jnp.ones(input_size)
        self._y_mean, self._y_std = jnp.zeros(output_size), jnp.ones(output_size)
        self.affine_transform_y = lambda dist: AffineTransform(shift=self._y_mean, scale=self._y_std)

        # disable some stupid tensorflow probability warnings
        self._add_checktypes_logging_filter()

    def _compute_normalization_stats(self, x: jnp.ndarray, y: jnp.ndarray) -> None:
        # computes the empirical normalization stats and stores as private variables
        x, y = self._ensure2d_float(x, y)
        self._x_mean = jnp.mean(x, axis=0)
        self._y_mean = jnp.mean(y, axis=0)
        self._x_std = jnp.std(x, axis=0)
        self._y_std = jnp.std(y, axis=0)

    def _normalize_data(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        # normalized the given data with the normalization stats
        if y is None:
            x = self._ensure2d_float(x)
        else:
            x, y = self._ensure2d_float(x, y)
        x_normalized = (x - self._x_mean[None, :]) / (self._x_std[None, :] + eps)
        assert x_normalized.shape == x.shape
        if y is None:
            return x_normalized
        else:
            y_normalized = (y - self._y_mean[None, :]) / (self._y_std[None, :] + eps)
            assert y_normalized.shape == y.shape
            return x_normalized, y_normalized

    @staticmethod
    def _ensure2d_float(x: jnp.ndarray, y: Optional[jnp.ndarray] = None, dtype: jnp.dtype = jnp.float32):
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
            self._compute_normalization_stats(x_train, y_train)
            x_train, y_train = self._normalize_data(x_train, y_train)
            self.affine_transform_y = AffineTransform(shift=self._y_mean, scale=self._y_std)
        else:
            x_train, y_train = self._ensure2d_float(x_train, y_train)
        return x_train, y_train

    def _create_data_loader(self, x_data: jnp.ndarray, y_data: jnp.ndarray,
                            batch_size: int = 64, shuffle: bool = True,
                            infinite: bool = True) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        if shuffle:
            seed = int(jax.random.randint(self.rng_key, (1,), 0, 10**8))
            ds = ds.shuffle(batch_size * 4, seed=seed, reshuffle_each_iteration=True)
        if infinite:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = tfds.as_numpy(ds)
        return ds

    def fit(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
            y_eval: Optional[jnp.ndarray] = None, num_steps: Optional[int] = None, log_period: int = 1000):
        # check whether eval data has been passed
        evaluate = x_eval is not None or y_eval is not None
        assert not evaluate or x_eval.shape[0] == y_eval.shape[0]

        # prepare data and data loader
        x_train, y_train = self._preprocess_train_data(x_train, y_train)
        train_loader = self._create_data_loader(x_train, y_train, batch_size=self.data_batch_size)
        num_train_points = x_train.shape[0]

        # initialize attributes to keep track of during training
        num_steps = self.num_steps if num_steps is None else num_steps
        samples_cum_period = 0.0
        stats_list = []
        t_start_period = time.time()

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
                    eval_stats = self.eval(x_eval, y_eval, prefix='eval_')
                    stats_agg.update(eval_stats)
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

    @property
    def likelihood_std(self):
        return jax.nn.softplus(self._likelihood_std_raw)

    @likelihood_std.setter
    def likelihood_std(self, std: jnp.ndarray):
        self._likelihood_std_raw = tfp.math.softplus_inverse(std)

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = False) -> tfp.distributions.Distribution:
        return False

    def predict(self, x: jnp.ndarray, include_noise: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    def eval(self, x: jnp.ndarray, y: np.ndarray, prefix: str = '') -> Dict[str, jnp.ndarray]:
        x, y = self._ensure2d_float(x, y)
        pred_dist = self.predict_dist(x, include_noise=True)
        nll = - jnp.mean(pred_dist.log_prob(y))
        rmse = jnp.sqrt(jnp.mean(jnp.sum((pred_dist.mean - y)**2, axis=-1)))
        eval_stats = {'nll': nll, 'rmse': rmse}
        # add prefix to stats names
        eval_stats = {f'{prefix}{name}': val for name, val in eval_stats.items()}
        return eval_stats

    def plot_1d(self, x_train: jnp.ndarray, y_train: jnp.ndarray,
                domain_l: Optional[float] = None, domain_u: Optional[float] = None,
                true_fun: Optional[Callable] = None, title: Optional[str] = '', show: bool = True):
        assert self.output_size == self.input_size == 1

        # determine plotting domain
        x_min, x_max = jnp.min(x_train, axis=0), jnp.max(x_train, axis=0)
        width = x_max - x_min
        if domain_l is None:
            domain_l = x_min - 0.3 * width
        if domain_u is None:
            domain_u = x_max + 0.3 * width
        x_plot = jnp.linspace(domain_l, domain_u, 200).reshape((-1, 1))

        # make predictions
        pred_mean, pred_std = map(lambda arr: arr.flatten(), self.predict(x_plot))

        # draw the plot
        fig, ax = plt.subplots()
        ax.scatter(x_train, y_train, label='train points')
        if true_fun is not None:
            ax.plot(x_plot, true_fun(x_plot), label='true fun')
        ax.plot(x_plot, pred_mean, label='pred mean')
        ax.fill_between(x_plot.flatten(), pred_mean - pred_std, pred_mean + pred_std, alpha=0.3)

        if hasattr(self, 'predict_post_samples'):
            y_post_samples = self.predict_post_samples(x_plot)
            for y in y_post_samples:
                ax.plot(x_plot, y, linewidth=0.2, color='green')

        if title is not None:
            ax.set_title(title)
        ax.legend()
        if show:
            fig.show()

    def _add_checktypes_logging_filter(self):
        logger = logging.getLogger()
        class CheckTypesFilter(logging.Filter):
            def filter(self, record):
                return "check_types" not in record.getMessage()
        logger.addFilter(CheckTypesFilter())