import jax
import jax.numpy as jnp
import numpy as np

from sim_transfer.models.bnn_fsvgd import BNN_FSVGD


def key_iter():
    key = jax.random.PRNGKey(3453)
    while True:
        key, new_key = jax.random.split(key)
        yield new_key


key_iter = key_iter()

NUM_DIM_X = 1
NUM_DIM_Y = 2
num_train_points = 20 * 32

fun = lambda x: jnp.concatenate([jnp.sin(4 * x).reshape(-1, 1), jnp.cos(4 * x).reshape(-1, 1)], axis=-1)

domain_l, domain_u = np.array([-1.] * NUM_DIM_X), np.array([1.] * NUM_DIM_X)

# x_train = jax.random.uniform(next(key_iter), shape=(num_train_points, NUM_DIM_X), minval=-5, maxval=5)
x_train = jnp.linspace(-1, 1, num_train_points).reshape(num_train_points, NUM_DIM_X)
y_train = fun(x_train) + 0.1 * jax.random.normal(next(key_iter), shape=(x_train.shape[0], NUM_DIM_Y))

num_test_points = 100
x_test = jax.random.uniform(next(key_iter), shape=(num_test_points, NUM_DIM_X), minval=0., maxval=10.)
noise_std = 0.2
y_test = fun(x_test) + noise_std * jax.random.normal(next(key_iter), shape=(x_test.shape[0], NUM_DIM_Y))

bnn = BNN_FSVGD(NUM_DIM_X, NUM_DIM_Y, domain_l, domain_u, next(key_iter), num_train_steps=4000,
                data_batch_size=32, num_measurement_points=20, normalize_data=True, bandwidth_svgd=0.2,
                bandwidth_gp_prior=0.2, hidden_layer_sizes=[64, 64, 64], num_particles=10,
                hidden_activation=jax.nn.swish, likelihood_std=noise_std)
for i in range(2):
    bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
    if NUM_DIM_X == 1:
        bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 2000}')
