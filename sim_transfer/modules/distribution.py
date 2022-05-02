import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


class AffineTransform:
    """Y = transform(X) = normalization_std @ X + normalization_mean"""

    def __init__(self, shift: jnp.array, scale: jnp.array):
        self.shift = shift
        self.scale = scale

        shift = tfp.bijectors.Shift(self.shift)
        if jnp.size(self.scale) == 1:
            scale = tfp.bijectors.Scale(self.scale)
        else:
            scale = tfp.bijectors.ScaleMatvecDiag(self.scale)
        self.transform = tfp.bijectors.Chain([shift, scale])

    def __call__(self, base_dist: tfp.distributions.Distribution) -> tfp.distributions.TransformedDistribution:
        # Transform distribution to access `log_prob` and `sample` methods
        transformed_dist = self.transform(base_dist)

        # Fill in mean, stddev and variance methods
        if callable(base_dist.mean):
            mean, stddev, var = base_dist.mean(), base_dist.stddev(), base_dist.variance()
        else:
            mean, stddev, var = base_dist.mean, base_dist.stddev, base_dist.variance
        transformed_dist.mean = self.transform(mean)
        transformed_dist.stddev = jnp.exp(jnp.log(stddev) + jnp.log(self.scale))
        transformed_dist.variance = jnp.exp(jnp.log(var) + 2. * jnp.log(self.scale))
        return transformed_dist