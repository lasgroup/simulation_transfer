import unittest
import jax
import jax.numpy as jnp
import optax
import pytest

from tensorflow_probability.substrates import jax as tfp

from sim_transfer.modules import Dense, MLP, SequentialModule, BatchedMLP
from sim_transfer.modules.util import tree_unstack, tree_stack, find_root_1d
from sim_transfer.modules.data_loader import DataLoader
from sim_transfer.modules.distribution import AffineTransform


class TestDenseParametrized(unittest.TestCase):

    def setUp(self) -> None:
        key1, key2 = jax.random.split(jax.random.PRNGKey(234), 2)
        self.fun = lambda x: 2 * x + 0.5
        self.x_train = jax.random.uniform(key1, shape=(20, 1), minval=-5, maxval=5)
        self.y_train = self.fun(self.x_train) + \
                       0.01 * jax.random.normal(key2, shape=self.x_train.shape)
        self.x_test = jnp.linspace(-5, 5, num=200).reshape((-1, 1))

    def test_basic_learning(self):
        key= jax.random.split(jax.random.PRNGKey(2567))
        dense = Dense(input_size=1, output_size=1, rng_key=key)

        optim = optax.adam(learning_rate=0.1)
        opt_state = optim.init(dense.params)
        params = dense.params

        def loss(params, x, y):
            return jnp.mean((dense.forward(x, params) - y) ** 2)

        for i in range(100):
            loss_value, grads = jax.value_and_grad(loss)(params, self.x_train, self.y_train)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        dense.params = params
        y_test = self.fun(self.x_test)
        y_pred = dense(self.x_test)
        mse = jnp.mean((y_test - y_pred)**2)
        assert mse < 0.1

    def test_basic_learning_vectorized(self):
        key = jax.random.split(jax.random.PRNGKey(2567))
        dense = Dense(input_size=1, output_size=1, rng_key=key)

        optim = optax.adam(learning_rate=0.1)
        opt_state = optim.init(dense.param_vector)
        param_vec = dense.param_vector

        @jax.jit
        def loss(param_vec, x, y):
            return jnp.mean((dense.forward_vec(x, param_vec) - y) ** 2)

        for i in range(100):
            loss_value, grads = jax.value_and_grad(loss)(param_vec, self.x_train, self.y_train)
            updates, opt_state = optim.update(grads, opt_state, param_vec)
            param_vec = optax.apply_updates(param_vec, updates)

        dense.param_vector = param_vec
        y_test = self.fun(self.x_test)
        y_pred = dense(self.x_test)
        mse = jnp.mean((y_test - y_pred) ** 2)
        assert mse < 0.1

    def test_params_getter_setter(self):
        key = jax.random.PRNGKey(324)
        dense = Dense(input_size=3, output_size=4, rng_key=key)

        w_inital = dense.w
        params1 = dense.params
        dense.w = dense.w**2
        dense.params = params1
        assert jnp.allclose(dense.w, w_inital)

    def test_params_getter_setter_vec(self):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(324), 3)
        dense = Dense(input_size=3, output_size=4, rng_key=key3)

        w_inital = dense.w
        param_vector1 = dense.param_vector
        dense.w = dense.w**2
        dense.param_vector = param_vector1
        assert jnp.allclose(dense.w, w_inital)


class TestSequential(unittest.TestCase):

    def setUp(self) -> None:
        def key_iter():
            key = jax.random.PRNGKey(324)
            while True:
                key, new_key = jax.random.split(key)
                yield new_key
        self.key_iter = key_iter()

        self.model = SequentialModule([
            Dense(1, 32, next(self.key_iter)),
            jax.nn.leaky_relu,
            Dense(32, 32, next(self.key_iter)),
            jax.nn.leaky_relu,
            Dense(32, 1, next(self.key_iter))
        ])

    def test_init_consistency(self):
        key = jax.random.PRNGKey(4564)
        params_init1 = self.model.init_params(rng_key=key, inplace=False)
        params_init2 = self.model.init_params(rng_key=key, inplace=False)
        assert all([jnp.allclose(p1, p2)
                    for p1, p2 in zip(jax.tree_leaves(params_init1), jax.tree_leaves(params_init2))])
        params_init_vec = self.model.get_init_param_vec(rng_key=key)
        self.model.init_params(rng_key=key, inplace=True)
        assert jnp.allclose(self.model.param_vector, params_init_vec)

    def test_params_getter_setter(self):
        ws_initial = [self.model.submodules_parametrized[i].w for i in range(3)]
        params1 = self.model.params
        for i in range(2):
            self.model.submodules_parametrized[i].w = self.model.submodules_parametrized[i].w ** 2
            assert not jnp.allclose(self.model.submodules_parametrized[i].w, ws_initial[i])
        self.model.params = params1
        for i in range(2):
            assert jnp.allclose(self.model.submodules_parametrized[i].w, ws_initial[i])

    def test_params_getter_setter_vectorized(self):
        ws_initial = [self.model.submodules_parametrized[i].w for i in range(3)]
        param_vec1 = self.model.param_vector
        for i in range(2):
            self.model.submodules_parametrized[i].w = self.model.submodules_parametrized[i].w ** 2
            assert not jnp.allclose(self.model.submodules_parametrized[i].w, ws_initial[i])
        self.model.param_vector = param_vec1
        for i in range(2):
            assert jnp.allclose(self.model.submodules_parametrized[i].w, ws_initial[i])

    def test_vec_to_params_bijection(self):
        param_vec = self.model.param_vector
        params = self.model._vec_to_params(param_vec)
        assert all([jnp.allclose(p1, p2) for p1, p2, in
                    zip(jax.tree_leaves(params), jax.tree_leaves(self.model.params))])


class TestMLP(unittest.TestCase):

    def setUp(self) -> None:
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(324), 3)
        self.key1 = key1
        self.fun = lambda x: 2 * x + 2 * jnp.sin(2 * x)
        self.x_train = jax.random.uniform(key3, shape=(40, 1), minval=-5, maxval=5)
        self.y_train = self.fun(self.x_train) + 0.1 * jax.random.normal(key3, shape=self.x_train.shape)

    def test_basic_learning_param(self):
        # test basic training loss convergence when using the pytree parameters
        optim = optax.adam(learning_rate=1e-3)
        model = MLP(1, 1, hidden_layer_sizes=[32, 32, 32], rng_key=self.key1)
        opt_state = optim.init(model.params)
        params = model.params

        def loss(params, x, y):
            return jnp.mean((model.forward(x, params) - y) ** 2)

        @jax.jit
        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(loss)(params, self.x_train, self.y_train)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for i in range(5000):
            params, opt_state, loss_value = step(params, opt_state)
            if i % 1000 == 0:
                print(i, loss_value)
        model.params = params

        pred1 = model(self.x_train)
        pred2 = model.forward(self.x_train, params)
        assert jnp.allclose(pred1, pred2)
        assert jnp.mean((pred1 - self.y_train) ** 2) < 0.1

    def test_basic_learning_param_vec(self):
        # test basic training loss convergence when using the vectorized parameters
        optim = optax.adam(learning_rate=1e-3)
        model = MLP(1, 1, hidden_layer_sizes=[32, 32, 32], rng_key=self.key1)
        opt_state = optim.init(model.param_vector)
        param_vec = model.param_vector

        def loss(param_vec, x, y):
            return jnp.mean((model.forward_vec(x, param_vec) - y) ** 2)

        @jax.jit
        def step(param_vec, opt_state):
            loss_value, grads = jax.value_and_grad(loss)(param_vec, self.x_train, self.y_train)
            updates, opt_state = optim.update(grads, opt_state, param_vec)
            param_vec = optax.apply_updates(param_vec, updates)
            return param_vec, opt_state, loss_value

        for i in range(5000):
            param_vec, opt_state, loss_value = step(param_vec, opt_state)
            if i % 1000 == 0:
                print(i, loss_value)
        model.param_vector = param_vec

        pred_vec = model.forward_vec(self.x_train, param_vec)
        pred = model(self.x_train)
        assert jnp.allclose(pred_vec, pred)
        assert jnp.mean((pred - self.y_train) ** 2) < 0.1


class TestBatchedMLP(unittest.TestCase):

    def test_batched_shapes(self):
        key_model, key_x = jax.random.split(jax.random.PRNGKey(7644), 2)
        num_modules = 5
        model = BatchedMLP(4, 3, hidden_layer_sizes=[8], num_batched_modules=num_modules,
                           rng_key=key_model)

        # check param shape
        num_params = 4 * 8 + 8 + 8 * 3 + 3
        assert model.param_vectors_stacked.shape == (num_modules, num_params)

        # check output shape
        x = jax.random.uniform(key_x, shape=(7, 4), minval=-5, maxval=5)
        y_pred1 = model(x)
        assert y_pred1.shape == (num_modules, 7, 3)

        # check that if we tile the inputs, we get the same
        x_tiled = jnp.repeat(x[None, :, :], repeats=5, axis=0)
        y_pred2 = model(x_tiled)
        assert jnp.allclose(y_pred1, y_pred2)

    def test_init(self):
        key_model, key_init = jax.random.split(jax.random.PRNGKey(345), 2)
        num_modules = 5
        model = BatchedMLP(4, 3, hidden_layer_sizes=[12, 12], num_batched_modules=num_modules,
                           rng_key=key_model)
        init_vec1 = model.get_init_param_vec_stacked(key_init)
        init_vec2 = model.get_init_param_vec_stacked(key_init)
        assert jnp.allclose(init_vec1, init_vec2)

        model.param_vectors_stacked = init_vec1
        assert jnp.allclose(init_vec1, model.param_vectors_stacked)

    def test_params_output_consistency(self):
        key_model, key_x = jax.random.split(jax.random.PRNGKey(456), 2)
        num_modules = 3
        model = MLP(1, 1, hidden_layer_sizes=[12,], rng_key=key_model)
        model_batched = BatchedMLP(1, 1, hidden_layer_sizes=[12,], num_batched_modules=num_modules,
                           rng_key=key_model)
        model_batched.param_vectors_stacked = jnp.stack([model.param_vector for _ in range(num_modules)], axis=0)

        x = jax.random.uniform(key_x, shape=(10, 1), minval=-5, maxval=5)
        y = 2 * x

        # check that the normal and the batch mlp output the same
        y_pred_batched = model_batched(x)
        y_pred = model(x)
        assert jnp.allclose(y_pred_batched[0], y_pred)
        assert jnp.allclose(y_pred_batched[0], y_pred_batched[1])

    def test_vec_to_params_consistency(self):
        key_model, key_init = jax.random.split(jax.random.PRNGKey(345), 2)
        num_modules = 2
        model = BatchedMLP(4, 3, hidden_layer_sizes=[12, 12], num_batched_modules=num_modules,
                           rng_key=key_model)
        p_vec1 = model.param_vectors_stacked
        params_stacked1 = model.params_stacked
        model.param_vectors_stacked = 2 * model.param_vectors_stacked
        model.params_stacked = params_stacked1
        assert jnp.allclose(p_vec1, model.param_vectors_stacked)


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

    @pytest.mark.parametrize(['batch_size', 'shuffle'], [[3, True], [5, True], [3, False]])
    @staticmethod
    def test_goes_through_all_data(batch_size: int, shuffle: bool):
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

    @pytest.mark.parametrize('drop_last', [True, False])
    @staticmethod
    def test_key_and_shape_consistency(drop_last: bool):
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(234), 3)
        x_data = jax.random.uniform(key1, (10, 8, 3))
        y_data = jax.random.uniform(key2, (10, 4))

        data_loader1 = DataLoader(x_data, y_data, rng_key=key3, shuffle=True, batch_size=4, drop_last=False)
        data_loader2 = DataLoader(x_data, y_data, rng_key=key3, shuffle=True, batch_size=4, drop_last=False)

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


if __name__ == '__main__':
    pytest.main()