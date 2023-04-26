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

    def test_sample_datasets_shapes(self):
        x_train, y_train, x_test, y_test = self.env.sample_datasets(
            self.rng_key, num_samples_train=5, num_samples_test=10, param_mode='random')
        assert x_train.shape == (5, self.env.input_size)
        assert y_train.shape == (5, self.env.output_size)
        assert x_test.shape == (10, self.env.input_size)
        assert y_test.shape == (10, self.env.output_size)

    def test_sample_datasets_seed_consistency(self):
        data_arrays1 = self.env.sample_datasets(
            self.rng_key, num_samples_train=5, num_samples_test=10, param_mode='random')
        data_arrays2 = self.env.sample_datasets(
            self.rng_key, num_samples_train=5, num_samples_test=10, param_mode='random')
        assert all([jnp.allclose(arr1, arr2) for arr1, arr2 in zip(data_arrays1, data_arrays2)])



if __name__ == '__main__':
    unittest.main()
