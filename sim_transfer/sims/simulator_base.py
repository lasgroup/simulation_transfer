from typing import Callable, Optional

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap, random
from tensorflow_probability.substrates import jax as tfp

from sim_transfer.sims.dynamics_models import Pendulum, PendulumParams


class FunctionSimulator:

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        raise NotImplementedError


class GaussianProcessSim(FunctionSimulator):

    def __init__(self, input_size: int = 1, output_size: int = 1, output_scale: float = 1.0, length_scale: float = 1.0,
                 mean_fn: Optional[Callable] = None):
        """ Samples functions from a Gaussian Process (GP) with SE kernel
        Args:
            input_size: dimensionality of the inputs
            output_scale: output_scale of the SE kernel (coincides with the std of the GP prior)
            length_scale: lengthscale of the SE kernel
            mean_fn (optional): mean function of the GP. If None, uses a zero mean.
        """
        super().__init__(input_size=input_size, output_size=output_size)

        if mean_fn is None:
            # use a zero mean by default
            self.mean_fn = lambda x: jnp.zeros((x.shape[0],))
        else:
            self.mean_fn = mean_fn

        self.output_scale = output_scale
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=length_scale)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """ Samples functions from a Gaussian Process (GP) with SE kernel
            Args:
                x: index/measurement points of size (n, input_size)
                num_samples: number of samples to draw
                rng_key: random number generator key
        """
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        gp = tfd.GaussianProcess(kernel=self.kernel, index_points=x, jitter=1e-4)
        keys = random.split(rng_key, self.output_size)
        f_samples = vmap(gp.sample, in_axes=(None, 0), out_axes=2)(num_samples, keys)
        f_samples *= self.output_scale
        assert f_samples.shape == (num_samples, x.shape[0], self.output_size)
        return f_samples


class SinusoidsSim(FunctionSimulator):

    def __init__(self, input_size: int = 1, output_size: int = 1):
        assert input_size == 1, 'only 1 dimensional inputs are supported'
        assert output_size in [1, 2], 'only 1 or 2-dimensional outputs are supported'
        super().__init__(input_size=input_size, output_size=output_size)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)
        freq = jax.random.uniform(key1, shape=(num_samples,), minval=1.7, maxval=2.3)
        amp = 2 + 0.4 * jax.random.normal(key2, shape=(num_samples,))
        slope = 2 + 0.3 * jax.random.normal(key3, shape=(num_samples,))
        f = amp[:, None, None] * jnp.sin(freq[:, None, None] * x) + slope[:, None, None] * x
        if self.output_size == 2:
            freq2 = jax.random.uniform(key4, shape=(num_samples,), minval=1.3, maxval=1.7)
            f2 = amp[:, None, None] * jnp.cos(freq2[:, None, None] * x) - slope[:, None, None] * x
            f = jnp.concatenate([f, f2], axis=-1)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f


class QuadraticSim(FunctionSimulator):
    def __init__(self):
        super().__init__(input_size=1, output_size=1)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        ks = jax.random.uniform(rng_key, shape=(num_samples,), minval=0.9, maxval=1.1)
        f = ks[:, None, None] * (x - 2) ** 2
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f


class PendulumSim(FunctionSimulator):
    def __init__(self, h: float = 0.01, upper_bound: PendulumParams | None = None,
                 lower_bound: PendulumParams | None = None):
        super().__init__(input_size=3, output_size=2)
        self.model = Pendulum(h=h)
        if upper_bound is None:
            upper_bound = PendulumParams(m=jnp.array(.5), l=jnp.array(.5), g=jnp.array(5.0), nu=jnp.array(0.0))
        if lower_bound is None:
            upper_bound = PendulumParams(m=jnp.array(1.5), l=jnp.array(1.5), g=jnp.array(15.0), nu=jnp.array(1.0))
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        keys = random.split(rng_key, num_samples)
        params = vmap(self.model.sample_params, in_axes=(0, None, None))(keys, self.upper_bound, self.lower_bound)

        def batched_fun(z, params):
            x, u = z[..., :2], z[..., 2:]
            return vmap(self.model.next_step, in_axes=(0, 0, None))(x, u, params)

        f = vmap(batched_fun, in_axes=(None, 0))(x, params)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    key = jax.random.PRNGKey(984)
    # sim = GaussianProcessSim(input_size=1, output_scale=3.0, mean_fn=lambda x: 2 * x)
    sim = SinusoidsSim()
    x_plot = jnp.linspace(-5, 5, 200).reshape((-1, 1))

    y_samples = sim.sample_function_vals(x_plot, 10, key)
    for y in y_samples:
        plt.plot(x_plot, y)
    plt.show()
