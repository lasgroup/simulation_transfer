import jax.numpy as jnp
import jax.random
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
import chex
from typing import Optional, Union
from tensorflow_probability.substrates.jax.internal import parameter_properties


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


class ParticleDistribution(tfd.MultivariateNormalDiag):
    def __init__(self,
                 particle_means: jnp.array,
                 aleatoric_stds: jnp.array,
                 calibration_alpha: Optional[Union[chex.Array, float]] = None,
                 base_seed: int = 0,
                 ):
        self.base_key = jax.random.PRNGKey(base_seed)
        self._particle_means = particle_means
        assert self._particle_means.ndim == 3  # particle size, batch size, dim
        self._num_particles, bs, self._dim = self._particle_means.shape

        if aleatoric_stds is None:
            aleatoric_stds = jnp.zeros(shape=(self._num_particles, bs, self._dim))
        # if aleatoric_stds.ndim == 1:
        # if aleatoric_std.shape != (self._num_particles, bs, self._dim), this corrects it
        aleatoric_stds = aleatoric_stds + jnp.zeros_like(particle_means)

        if calibration_alpha is None or isinstance(calibration_alpha, float):
            if calibration_alpha is None:
                calibration_alpha = 1.0
            calibration_alpha = jnp.ones(shape=(self._dim,)) * calibration_alpha

        self._calibration_alpha = calibration_alpha

        self._aleatoric_stds = aleatoric_stds
        scale_diag = self.total_stddev()
        super().__init__(
            loc=jnp.mean(self._particle_means, axis=0),
            scale_diag=scale_diag,
        )

    def total_stddev(self) -> chex.Array:
        # Total std is sqrt of variance of particles and mean of aleatoric stds
        eps_var = (jnp.std(self._particle_means, axis=0) * self._calibration_alpha) ** 2
        ale_var = jnp.mean(self._aleatoric_stds ** 2, axis=0)
        total_std = jnp.sqrt(eps_var + ale_var)
        return total_std

    def median(self) -> chex.Array:
        return jnp.median(self._particle_means, axis=0)

    @property
    def particle_means(self) -> chex.Array:
        return self._particle_means

    @property
    def particle_aleatoric_stds(self) -> chex.Array:
        return self._aleatoric_stds

    @property
    def raw_aleatoric_std(self) -> chex.Array:
        return jnp.mean(self._aleatoric_stds, axis=0)

    @property
    def raw_epistemic_std(self) -> chex.Array:
        return jnp.std(self._particle_means, axis=0)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        dict = tfd.MultivariateNormalDiag._parameter_properties(dtype, num_classes)
        return dict