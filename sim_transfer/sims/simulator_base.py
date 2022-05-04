import jax
import jax.numpy as jnp
from typing import Callable, Optional
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd


class FunctionSimulator:

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        raise NotImplementedError


class GaussianProcessSim(FunctionSimulator):

    def __init__(self, input_size: int = 1, output_scale: float = 1.0, length_scale: float = 1.0,
                 mean_fn: Optional[Callable] = None,
                 eps_k: float = 1e-5):
        """ Samples functions from a Gaussian Process (GP) with SE kernel
        Args:
            input_size: dimensionality of the inputs
            output_scale: output_scale of the SE kernel (coincides with the std of the GP prior)
            length_scale: lengthscale of the SE kernel
            mean_fn (optional): mean function of the GP. If None, uses a zero mean.
            eps_k: scale of the identity matrix to be added to the kernel matrix for numerical stability
        """
        super().__init__(input_size=input_size, output_size=1)

        if mean_fn is None:
            # use a zero mean by default
            self.mean_fn = lambda x: jnp.zeros((x.shape[0],))
        else:
            self.mean_fn = mean_fn

        self.output_scale = output_scale
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=length_scale)
        self.eps_k = eps_k

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        m = self.mean_fn(x).reshape((x.shape[0],))
        k = self.kernel.matrix(x, x)
        k += self.eps_k * jnp.eye(k.shape[0])  # add eps * I for numerical stability of the cholesky decomposition
        gp_noise = tfd.MultivariateNormalFullCovariance(jnp.zeros_like(m), k).sample(num_samples, seed=rng_key)
        f = m + self.output_scale * gp_noise
        f = jnp.expand_dims(f, axis=-1)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f


class SinusoidsSim(FunctionSimulator):

    def __init__(self):
        super().__init__(input_size=1, output_size=1)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        key1, key2, key3 = jax.random.split(rng_key, 3)
        freq = jax.random.uniform(key1, shape=(num_samples,), minval=1.7, maxval=2.3)
        amp = 2 + 0.4 * jax.random.normal(key2, shape=(num_samples,))
        slope = 2 + 0.3 * jax.random.normal(key2, shape=(num_samples,))
        f = amp[:, None, None] * jnp.sin(freq[:, None, None] * x) + slope[:, None, None] * x
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f



if __name__ == '__main__':
    from matplotlib import pyplot as plt

    key = jax.random.PRNGKey(984)
    #sim = GaussianProcessSim(input_size=1, output_scale=3.0, mean_fn=lambda x: 2 * x)
    sim = SinusoidsSim()
    x_plot = jnp.linspace(-5, 5, 200).reshape((-1, 1))

    y_samples = sim.sample_function_vals(x_plot, 10, key)
    for y in y_samples:
        plt.plot(x_plot, y)
    plt.show()