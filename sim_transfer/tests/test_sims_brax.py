import unittest
import jax
import jax.numpy as jnp
from sim_transfer.sims.simulators_brax import RandomInvertedPendulumEnv


class TestRandomInvertedPendulumEnv(unittest.TestCase):

    def setUp(self):
        self.env = RandomInvertedPendulumEnv()
        self.rng_key = jax.random.PRNGKey(0)
        self.x = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])
        self.num_samples = 3

    def test_seed_consistency_predict_next_random_vmap(self):
        keys = jax.random.split(self.rng_key, self.num_samples)
        result1 = self.env._predict_next_random_vmap(keys, self.x)
        result2 = self.env._predict_next_random_vmap(keys, self.x)
        self.assertTrue(jnp.allclose(result1, result2))

    def test_seed_consistency_sample_function_vals(self):
        result1 = self.env.sample_function_vals(self.x, self.num_samples, self.rng_key)
        result2 = self.env.sample_function_vals(self.x, self.num_samples, self.rng_key)
        self.assertTrue(jnp.allclose(result1, result2))

    def test_consistency_sample_function_vals_wrt_x(self):
        x = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        result = self.env.sample_function_vals(x, self.num_samples, self.rng_key)
        for i in range(self.num_samples):
            self.assertTrue(jnp.allclose(result[i, 0, :], result[i, 1, :]))


if __name__ == '__main__':
    unittest.main()
