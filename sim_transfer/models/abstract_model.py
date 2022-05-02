import jax
import jax.numpy as jnp
from typing import Optional
from sim_transfer.modules.util import RngKeyMixin


class AbstactRegressionModel(RngKeyMixin):

    def __init__(self, input_size: int, output_size: int, rng_key: jax.random.PRNGKey):
        super().__init__(rng_key)

        self.input_size = input_size
        self.output_size = output_size

        # initialize normalization stats to neutral elements
        self._x_mean, self._x_std = jnp.zeros(input_size), jnp.ones(input_size)
        self._y_mean, self._y_std = jnp.zeros(output_size), jnp.ones(output_size)

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