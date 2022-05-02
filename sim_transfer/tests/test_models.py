import unittest
import jax
import jax.numpy as jnp

from sim_transfer.models.abstract_model import AbstactRegressionModel

class TestAbstractRegression(unittest.TestCase):

    def test_normalization(self):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(45645), 3)
        x_mean, x_std = jnp.array([1., -2.]), jnp.array([1., 5.])
        y_mean, y_std = jnp.array([5.0]), jnp.array([0.1])
        x_data = x_mean + x_std * jax.random.normal(key1, (100, 2))
        y_data = y_mean + y_std * jax.random.normal(key2, (100, 1))
        y_data = y_data.flatten()

        # check that normalization has no effect when we don't compute normalization stats and use the
        # default zero mean and std of 1
        model = AbstactRegressionModel(input_size=2, output_size=1, rng_key=key3)
        x = model._normalize_data(x_data)
        assert jnp.mean(jnp.linalg.norm(x - x_data, axis=0)) <= 1e-4

        x, y = model._normalize_data(x_data, y_data)
        assert jnp.mean(jnp.linalg.norm(x - x_data, axis=0)) <= 1e-4
        assert jnp.mean(jnp.linalg.norm(y - y_data.reshape((-1, 1)), axis=0)) <= 1e-4

        # now check that the data is normalized
        model._compute_normalization_stats(x_data, y_data)
        x, y = model._normalize_data(x_data, y_data)
        assert jnp.linalg.norm(jnp.mean(x, axis=0)) < 1e-1
        assert jnp.linalg.norm(jnp.std(x, axis=0) - 1.0) < 1e-1
        assert jnp.linalg.norm(jnp.mean(y, axis=0)) < 1e-1
        assert jnp.linalg.norm(jnp.std(y, axis=0) - 1.0) < 1e-1

    def test_data_loader_epoch(self):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(45645), 3)
        x_data = jnp.arange(0, 30).reshape((-1, 1))
        y_data = jnp.arange(0, 30).reshape((-1, 1))

        model1 = AbstactRegressionModel(input_size=2, output_size=1, rng_key=key3)
        data_loader1 = model1._create_data_loader(x_data, y_data, batch_size=7, shuffle=True, infinite=False)

        model2 = AbstactRegressionModel(input_size=2, output_size=1, rng_key=key3)
        data_loader2 = model2._create_data_loader(x_data, y_data, batch_size=7, shuffle=True, infinite=False)

        x1_batch_list = []
        for (x1, y1), (x2, y2) in zip(data_loader1, data_loader2):
            assert jnp.allclose(x1, y2) and jnp.allclose(y1, y2)  # label consistency
            assert jnp.allclose(x1, x2)  # seed consistency
            x1_batch_list.append(x1)
        x1_cat = jnp.sort(jnp.concatenate(x1_batch_list), axis=0)
        assert jnp.allclose(x_data, x1_cat)  # check that it goes through all data points

    def test_data_loader_infinite(self):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(45645), 3)
        x_data = jnp.arange(0, 30).reshape((-1, 1))
        y_data = jnp.arange(0, 30).reshape((-1, 1))
        model = AbstactRegressionModel(input_size=2, output_size=1, rng_key=key3)
        data_loader = model._create_data_loader(x_data, y_data, batch_size=7, shuffle=True, infinite=True)

        x_batch_list = []
        for x, y in data_loader:
            assert jnp.allclose(x, x)
            assert x.shape[0] == 7
            x_batch_list.append(x)
            if len(x_batch_list) >= 5:
                break
        assert set(jnp.concatenate(x_batch_list).flatten().tolist()) == set(x_data.flatten().tolist())


if __name__ == '__main__':
    unittest.main()