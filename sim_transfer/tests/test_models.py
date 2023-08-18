import unittest
import jax
import jax.numpy as jnp

from sim_transfer.models.abstract_model import AbstractRegressionModel
from sim_transfer.models.bnn_svgd import BNN_SVGD

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
        model = AbstractRegressionModel(input_size=2, output_size=1, rng_key=key3)
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

    def test_normalization_unnormalization(self):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(974), 3)
        x_data = jnp.array([1., -2.]) + jnp.array([1., 5.]) * jax.random.normal(key1, (10, 2))
        y_data = jnp.array([5.0]) + jnp.array([0.1]) * jax.random.normal(key2, (10, 1))

        model = AbstractRegressionModel(input_size=2, output_size=1, rng_key=key3)
        model._compute_normalization_stats(x_data, y_data)

        x1, y1 = model._unnormalize_data(*model._normalize_data(x_data, y_data))
        assert jnp.allclose(x_data, x1)
        assert jnp.allclose(y_data, y1)

        x2 = model._unnormalize_data(model._normalize_data(x_data))
        assert jnp.allclose(x_data, x2)

        y2 = model._unnormalize_y(model._normalize_y(y_data))
        assert jnp.allclose(y_data, y2)

    def test_setting_normalization_stats(self):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(897), 3)
        x_data = jnp.array([1.,]) + jnp.array([1.]) * jax.random.normal(key1, (10, 1))
        y_data = jnp.array([5.0, -3.0]) + jnp.array([0.1, 5.]) * jax.random.normal(key2, (10, 2))

        norm_stats = {'x_mean': jnp.array([1.]), 'x_std': jnp.array([2.]),
                      'y_mean': jnp.array([2.0, -2.0]), 'y_std': jnp.array([0.1, 5.])}
        model = AbstractRegressionModel(input_size=1, output_size=2, rng_key=key3,
                                        normalization_stats=norm_stats)

        x_norm = model._normalize_data(x_data, eps=1e-8)
        y_norm = model._normalize_y(y_data, eps=1e-8)

        assert jnp.allclose(x_norm, (x_data - norm_stats['x_mean']) / (norm_stats['x_std'] + 1e-8))
        assert jnp.allclose(y_norm, (y_data - norm_stats['y_mean']) / (norm_stats['y_std'] + 1e-8))

    def test_data_loader_epoch(self):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(45645), 3)
        x_data = jnp.arange(0, 30).reshape((-1, 1))
        y_data = jnp.arange(0, 30).reshape((-1, 1))

        model1 = AbstractRegressionModel(input_size=2, output_size=1, rng_key=key3)
        data_loader1 = model1._create_data_loader(x_data, y_data, batch_size=7, shuffle=True, infinite=False)

        model2 = AbstractRegressionModel(input_size=2, output_size=1, rng_key=key3)
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
        model = AbstractRegressionModel(input_size=2, output_size=1, rng_key=key3)
        data_loader = model._create_data_loader(x_data, y_data, batch_size=7, shuffle=True, infinite=True)

        x_batch_list = []
        for x, y in data_loader:
            assert jnp.allclose(x, x)
            assert x.shape[0] == 7
            x_batch_list.append(x)
            if len(x_batch_list) >= 5:
                break
        assert set(jnp.concatenate(x_batch_list).flatten().tolist()) == set(x_data.flatten().tolist())


class TestReinitialization(unittest.TestCase):

    def test_reinit_bnn_svgd_1(self):
        bnn_svgd = BNN_SVGD(input_size=1, output_size=1, rng_key=jax.random.PRNGKey(34534527))

        # check that predictions are consistent when not reinitialized
        x = jnp.linspace(-1, 1, 100).reshape((-1, 1))
        y1_mean, y1_std = bnn_svgd.predict(x)
        y2_mean, y2_std = bnn_svgd.predict(x)
        assert jnp.allclose(y1_mean, y2_mean) and jnp.allclose(y1_std, y2_std)

        # check that predictions are different when reinitialized
        bnn_svgd.reinit()
        y3_mean, y3_std = bnn_svgd.predict(x)
        assert (not jnp.allclose(y1_mean, y3_mean)) and (not jnp.allclose(y1_std, y3_std))

    def test_reinit_bnn_svgd_seed_consistency(self):
        bnn_svgd = BNN_SVGD(input_size=1, output_size=1, rng_key=jax.random.PRNGKey(34534527))

        # generate train and test data
        x_train = jnp.linspace(-1, 1, 10).reshape((-1, 1))
        y_train = jnp.sin(x_train)
        x_test = jnp.linspace(-1, 1, 100).reshape((-1, 1))

        # reinit model, train for 10 steps, then make predictions
        key_reinit = jax.random.PRNGKey(904352)
        bnn_svgd.reinit(rng_key=key_reinit)
        bnn_svgd.fit(x_train, y_train, num_steps=10)
        y1_mean, y1_std = bnn_svgd.predict(x_test)

        # repeat
        bnn_svgd.reinit(rng_key=key_reinit)
        bnn_svgd.fit(x_train, y_train, num_steps=10)
        y2_mean, y2_std = bnn_svgd.predict(x_test)

        # check that predictions are the same
        assert jnp.allclose(y1_mean, y2_mean) and jnp.allclose(y1_std, y2_std)

        # check that if we reinitialize with a differnt seed, we get different predictions
        bnn_svgd.reinit(rng_key=jax.random.PRNGKey(456456))
        bnn_svgd.fit(x_train, y_train, num_steps=10)
        y3_mean, y3_std = bnn_svgd.predict(x_test)
        assert (not jnp.allclose(y1_mean, y3_mean)) and (not jnp.allclose(y1_std, y3_std))


if __name__ == '__main__':
    unittest.main()