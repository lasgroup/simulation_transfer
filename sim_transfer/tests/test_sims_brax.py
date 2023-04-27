import unittest
import jax
import jax.numpy as jnp
from sim_transfer.sims.simulators_brax import RandomInvertedPendulumEnv, RandomInvertedDoublePendulumEnv

import pytest

rng_key = jax.random.PRNGKey(2342)
num_samples = 3

ENVS = [RandomInvertedPendulumEnv(), RandomInvertedDoublePendulumEnv()]
INPUTS = [jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]),  # input for RandomInvertedPendulumEnv
          jnp.array([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # input for RandomInvertedDoublePendulumEnv
                     [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.2, -0.5, 0.4]])
          ]

@pytest.mark.parametrize('env, x', zip(ENVS, INPUTS))
def test_seed_consistency_predict_next_random_vmap(env, x):
    keys = jax.random.split(rng_key, num_samples)
    result1 = env._predict_next_random_vmap(keys, x)
    result2 = env._predict_next_random_vmap(keys, x)
    assert jnp.allclose(result1, result2)

@pytest.mark.parametrize('env, x', zip(ENVS, INPUTS))
def test_seed_consistency_sample_function_vals(env, x):
    result1 = env.sample_function_vals(x, num_samples, rng_key)
    result2 = env.sample_function_vals(x, num_samples, rng_key)
    assert jnp.allclose(result1, result2)

@pytest.mark.parametrize('env, x', zip(ENVS, INPUTS))
def test_consistency_sample_function_vals_wrt_x(env, x):
    x_same = jnp.stack([x[0], x[0]], axis=0)
    result = env.sample_function_vals(x_same, num_samples, rng_key)
    for i in range(num_samples):
        assert jnp.allclose(result[i, 0, :], result[i, 1, :])

@pytest.mark.parametrize('env', ENVS)
def test_sample_datasets_shapes(env):
    x_train, y_train, x_test, y_test = env.sample_datasets(
        rng_key, num_samples_train=5, num_samples_test=10, param_mode='random')
    assert x_train.shape == (5, env.input_size)
    assert y_train.shape == (5, env.output_size)
    assert x_test.shape == (10, env.input_size)
    assert y_test.shape == (10, env.output_size)

@pytest.mark.parametrize('env', ENVS)
def test_sample_datasets_seed_consistency(env):
    data_arrays1 = env.sample_datasets(
        rng_key, num_samples_train=5, num_samples_test=10, param_mode='random')
    data_arrays2 = env.sample_datasets(
        rng_key, num_samples_train=5, num_samples_test=10, param_mode='random')
    assert all([jnp.allclose(arr1, arr2) for arr1, arr2 in zip(data_arrays1, data_arrays2)])


def test_angle_encoding_double_pendulum():
    env = RandomInvertedDoublePendulumEnv()
    x1 = jnp.array([0.5, 1., -2.5])
    x2 = env._decode_q(env._encode_q(x1))
    assert jnp.linalg.norm(x1-x2) < 1e-4


if __name__ == '__main__':
    pytest.main()
