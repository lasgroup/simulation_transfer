import unittest
import jax
import jax.numpy as jnp

from sim_transfer.sims.domain import HypercubeDomain

class TestHypercubeDomain(unittest.TestCase):

    def setUp(self):
        self.mid1d = jnp.array([-3])
        self.scale1d = jnp.array([10])
        self.domain1d = HypercubeDomain(lower=self.mid1d - self.scale1d, upper=self.mid1d + self.scale1d)

        self.mid2d = jnp.array([5., -5.])
        self.scale2d = jnp.array([10., 5.])
        self.domain2d = HypercubeDomain(lower=self.mid2d - self.scale2d, upper=self.mid2d + self.scale2d)

    def test_data_shapes(self):
        # test whether the generated samples have the correct shape
        key = jax.random.PRNGKey(2234)
        assert self.domain2d.sample_uniformly(key, sample_shape=15).shape == (15, 2)
        assert self.domain2d.sample_uniformly(key, sample_shape=(5, 3, 1)).shape == (5, 3, 1, 2)

    def test_boundaries_full_support_mode(self):
        key = jax.random.PRNGKey(790234)
        samples = self.domain1d.sample_uniformly(key, sample_shape=1000, support_mode='full')
        assert jnp.all((self.domain1d.lower < samples) & (samples < self.domain1d.upper))

        samples = self.domain2d.sample_uniformly(key, sample_shape=1000, support_mode='full')
        assert jnp.all((self.domain2d.lower < samples) & (samples < self.domain2d.upper))

    def test_boundaries_partial_support_mode(self):
        key = jax.random.PRNGKey(3456)
        samples = self.domain1d.sample_uniformly(key, sample_shape=1000, support_mode='partial')
        assert jnp.all((self.mid1d - 0.9 * self.scale1d < samples) & (samples < self.mid1d + 0.9 * self.scale1d))
        assert jnp.all(jnp.linalg.norm((samples - self.mid1d) / self.scale1d, ord=jnp.inf, axis=-1) > 0.05)

        cutoff_scalar = (1 - 0.1)**(1/2)
        samples = self.domain2d.sample_uniformly(key, sample_shape=1000, support_mode='partial')
        assert jnp.all((self.mid2d - cutoff_scalar * self.scale2d < samples) &
                       (samples < self.mid2d + cutoff_scalar * self.scale2d))
        assert jnp.all(jnp.linalg.norm((samples - self.mid2d) / self.scale2d, ord=jnp.inf, axis=-1) > 0.05)







if __name__ == '__main__':
    unittest.main()