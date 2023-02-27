import unittest
import jax
import jax.numpy as jnp
import optax
import pytest

from tensorflow_probability.substrates import jax as tfp

from sim_transfer.modules.nn_modules import BatchedMLP, MLP
from sim_transfer.modules.util import tree_unstack, tree_stack, find_root_1d
from sim_transfer.modules.data_loader import DataLoader
from sim_transfer.modules.distribution import AffineTransform
from sim_transfer.modules.util import mmd2


class TestBatchedMLP(unittest.TestCase):

    def test_batched_shapes(self):
        key_model, key_x = jax.random.split(jax.random.PRNGKey(7644), 2)
        num_modules = 5
        model = BatchedMLP(input_size=4, output_size=3, hidden_layer_sizes=[8],
                             num_batched_modules=num_modules, rng_key=key_model)

        # check param shape
        num_params = 4 * 8 + 8 + 8 * 3 + 3
        assert model.flatten_batch(model.param_vectors_stacked).shape == (num_modules, num_params)

        # check output shape
        x = jax.random.uniform(key_x, shape=(7, 4), minval=-5, maxval=5)
        y_pred1 = model(x)
        assert y_pred1.shape == (num_modules, 7, 3)

    def test_init(self):
        key_model, key_init = jax.random.split(jax.random.PRNGKey(345), 2)
        num_modules = 5
        model = BatchedMLP(input_size=4, output_size=3, hidden_layer_sizes=[12, 12],
                           num_batched_modules=num_modules, rng_key=key_model)
        init_vec1 = model.get_init_param_vec_stacked(key_init)
        init_vec2 = model.flatten_batch(model.get_init_param_vec_stacked(key_init))
        assert jnp.allclose(model.flatten_batch(init_vec1), model.flatten_batch(init_vec2))

        model.param_vectors_stacked = init_vec1
        assert jnp.allclose(model.flatten_batch(init_vec1), model.flatten_batch(model.param_vectors_stacked))

    def test_params_output_consistency(self):
        key_model, key_x = jax.random.split(jax.random.PRNGKey(456), 2)
        num_modules = 3
        model = MLP(1, 1, hidden_layer_sizes=[12,])
        params = model.init(key_model, jnp.ones(shape=(1,)))
        model_batched = BatchedMLP(input_size=1, output_size=1, hidden_layer_sizes=[12],
                                   num_batched_modules=num_modules, rng_key=key_model)
        model_batched.param_vectors_stacked = tree_stack([params for _ in range(num_modules)])['params']

        x = jax.random.uniform(key_x, shape=(10, 1), minval=-5, maxval=5)
        y = 2 * x

        # check that the normal and the batch mlp output the same
        y_pred_batched = model_batched(x)
        y_pred = model.apply(params, x)
        assert jnp.allclose(y_pred_batched[0], y_pred)
        assert jnp.allclose(y_pred_batched[0], y_pred_batched[1])

    def test_vec_to_params_consistency(self):
        key_model, key_init = jax.random.split(jax.random.PRNGKey(345), 2)
        num_modules = 2
        model = BatchedMLP(input_size=4, output_size=3, hidden_layer_sizes=[12, 12],
                             num_batched_modules=num_modules, rng_key=key_model)
        p_vecs1 = model.param_vectors_stacked
        p_vecs2 = model.unravel_batch(model.flatten_batch(p_vecs1))
        assert all([jnp.allclose(param1, param2) for param1, param2 in zip(jax.tree_leaves(p_vecs1), jax.tree_leaves(p_vecs2))])


class TestUtil(unittest.TestCase):

    def test_tree_stack_unstack_consistency(self):
        key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(7385), 4)
        tree1 = [{'a': jax.random.normal(key1, (2, 4)), 'b': jax.random.normal(key2, (4,))},
                 {'c': 1.0}]
        tree2 = [{'a': jax.random.normal(key3, (2, 4)), 'b': jax.random.normal(key4, (4, ))},
                 {'c': 2.0}]
        stacked_tree = tree_stack([tree1, tree2])
        tree1_after, tree2_after = tree_unstack(stacked_tree)
        assert jnp.allclose(tree2[0]['a'], tree2_after[0]['a'])
        assert jnp.allclose(tree1[1]['c'], tree1_after[1]['c'])


class TestDataLoader(unittest.TestCase):

    def test_goes_through_all_data(self, batch_size: int = 3, shuffle: bool = True):
        key = jax.random.PRNGKey(234)
        num_data_points = 20
        x_data = jnp.arange(num_data_points).reshape((-1, 1))
        y_data = jnp.arange(num_data_points).reshape((-1, 1))

        data_loader = DataLoader(x_data, y_data, rng_key=key, shuffle=shuffle, batch_size=batch_size, drop_last=False)
        x_list = []
        y_list = []
        for i, (x, y) in enumerate(data_loader):
            x_list.append(x)
            y_list.append(x)
            assert jnp.allclose(x, y)
            assert (x.shape[0] == batch_size and y.shape[0] == batch_size) or \
                   (num_data_points % batch_size > 0 and i == num_data_points // batch_size)
        x_cat = jnp.concatenate(x_list, axis=0)
        y_cat = jnp.concatenate(y_list, axis=0)
        sorted_idx = jnp.argsort(x_cat, axis=0).flatten()
        assert jnp.allclose(x_cat[sorted_idx], x_data)
        assert jnp.allclose(y_cat[sorted_idx], y_data)

    def test_key_and_shape_consistency(self, drop_last: bool = True):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(234), 3)
        x_data = jax.random.uniform(key1, (10, 8, 3))
        y_data = jax.random.uniform(key2, (10, 4))

        data_loader1 = DataLoader(x_data, y_data, rng_key=key3, shuffle=True, batch_size=4, drop_last=drop_last)
        data_loader2 = DataLoader(x_data, y_data, rng_key=key3, shuffle=True, batch_size=4, drop_last=drop_last)

        cum_batch_sizes = 0
        for (x1, y1), (x2, y2) in zip(data_loader1, data_loader2):
            assert jnp.allclose(x1, x2) and jnp.allclose(y1, y2)
            assert x1.shape[1:] == (8, 3)
            assert y2.shape[1:] == (4,)
            assert x1.shape[0] == y1.shape[0] == x1.shape[0] == y2.shape[0]
            cum_batch_sizes += x1.shape[0]

        assert (drop_last and cum_batch_sizes == 8) or ((not drop_last) and cum_batch_sizes == 10)

    def test_multiple_epochs(self):
        key = jax.random.PRNGKey(234)
        x_data = jnp.arange(20).reshape((-1, 1))
        y_data = jnp.arange(20).reshape((-1, 1))

        data_loader = DataLoader(x_data, y_data, rng_key=key, shuffle=True, batch_size=7, drop_last=False)

        # epoch 1
        x_cat1, y_cat1 = list(map(lambda l: jnp.concatenate(l), list(zip(*data_loader))))
        #epoch 2
        x_cat2, y_cat2 = list(map(lambda l: jnp.concatenate(l), list(zip(*data_loader))))

        assert not jnp.allclose(x_cat1, x_cat2)
        assert not jnp.allclose(y_cat1, y_cat2)

        sorted_idx = jnp.argsort(x_cat1, axis=0).flatten()
        assert jnp.allclose(x_cat1[sorted_idx], y_cat1[sorted_idx])
        sorted_idx = jnp.argsort(x_cat2, axis=0).flatten()
        assert jnp.allclose(x_cat2[sorted_idx], y_cat2[sorted_idx])


class TestAffineTransformedDist(unittest.TestCase):

    def test_transformation(self):
        mean = jnp.array([[1.0, 2.0], [-1.0, 1.0], [1.0, 1.0]])
        std = jnp.array([[1.0, 1.4], [0.1, 4.2], [2., 0.5]])
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)

        shift = jnp.array([15., -6])
        scale = jnp.array([0.5, 3.])
        dist_tansf = AffineTransform(shift, scale)(dist)

        assert jnp.allclose(dist.mean() * scale + shift, dist_tansf.mean)
        assert jnp.allclose(dist.stddev() * scale, dist_tansf.stddev)
        assert jnp.allclose(dist.variance() * scale**2, dist_tansf.variance)


class TestRootFind1d(unittest.TestCase):

    def test_root_x3(self):
        f = lambda x: (x-3.)**3
        r =  find_root_1d(f)
        assert jnp.allclose(r, 3., atol=1e-4)

    def test_tanh(self):
        f = lambda x: 0.5 * jnp.tanh(20 * (x + 0.05))
        r =  find_root_1d(f)
        assert jnp.allclose(r, -.05, atol=1e-4)


class TestMMD(unittest.TestCase):

    def setUp(self) -> None:
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(567), 3)
        self.x = jax.random.normal(key1, shape=(20, 2))
        self.y = jax.random.normal(key1, shape=(30, 2))
        self.z = 5 + 2 * jax.random.normal(key1, shape=(25, 2))
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()

    def test_mmd_of_same_dist_smaller1(self):
        mmd = mmd2(self.x, self.x, self.kernel)
        assert mmd < 1e-5

    def test_mmd_of_same_dist_smaller2(self):
        for d in [True, False]:
            assert mmd2(self.x, self.y, self.kernel, include_diag=d) < mmd2(self.x, self.z, self.kernel, include_diag=d)

    def test_multiple_lengthscales(self):
        for include_diag in [True, False]:
            kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=[0.1, 2.])
            mmd_joint = mmd2(self.x, self.y, kernel, include_diag=include_diag)
            assert mmd_joint.shape == (2,)

            kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=0.1)
            assert jnp.array_equal(mmd2(self.x, self.y, kernel, include_diag=include_diag), mmd_joint[0])

            kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=2.)
            assert jnp.array_equal(mmd2(self.x, self.y, kernel, include_diag=include_diag), mmd_joint[1])


if __name__ == '__main__':
    pytest.main()