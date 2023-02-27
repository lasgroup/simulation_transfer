import jax
import jax.numpy as jnp
from jax import vmap

from sim_transfer.models.bnn_fsvgd_sim_prior import BNN_FSVGD_Sim_Prior
from sim_transfer.sims import PendulumSim
from sim_transfer.sims.dynamics_models import PendulumParams, Pendulum


def key_iter():
    key = jax.random.PRNGKey(7644)
    while True:
        key, new_key = jax.random.split(key)
        yield new_key


key_iter = key_iter()
x_dim = 2
u_dim = 1
time_step = 0.1

NUM_DIM_X = x_dim + u_dim
NUM_DIM_Y = x_dim

num_train_points = 10 * 32

# Here we define the true model
true_pendulum_params = PendulumParams(m=jnp.array(1.0), l=jnp.array(1.0), g=jnp.array(9.81), nu=jnp.array(0.1))
pendulum_model = Pendulum(h=time_step)

# Here we define prior
lower_b_params = PendulumParams(m=jnp.array(0.8), l=jnp.array(0.7), g=jnp.array(8.0), nu=jnp.array(0.0))
upper_b_params = PendulumParams(m=jnp.array(1.2), l=jnp.array(1.3), g=jnp.array(11.9), nu=jnp.array(1.0))
sim = PendulumSim(h=time_step, lower_bound=lower_b_params, upper_bound=upper_b_params)


def fun(z):
    assert z.shape == (NUM_DIM_X,)
    x, u = z[:x_dim], z[x_dim:]
    return pendulum_model.ode(x, u, true_pendulum_params)


x_train = jax.random.uniform(next(key_iter), shape=(num_train_points, NUM_DIM_X), minval=-5, maxval=5)
y_train = vmap(fun)(x_train) + 0.1 * jax.random.normal(next(key_iter), shape=(x_train.shape[0], NUM_DIM_Y))

num_test_points = 100
x_test = jax.random.uniform(next(key_iter), shape=(num_test_points, NUM_DIM_X), minval=-5, maxval=5)
y_test = vmap(fun)(x_test) + 0.1 * jax.random.normal(next(key_iter), shape=(x_test.shape[0], NUM_DIM_Y))
domain_l, domain_u = -5 * jnp.ones(shape=(NUM_DIM_X)), 5 * jnp.ones(shape=(NUM_DIM_X))

# sim = GaussianProcessSim(input_size=1, output_scale=3.0, mean_fn=lambda x: 2 * x)
# sim = SinusoidsSim(input_size=1, output_size=NUM_DIM_Y)
# sim = PendulumSim()
bnn = BNN_FSVGD_Sim_Prior(NUM_DIM_X, NUM_DIM_Y, domain_l, domain_u, rng_key=next(key_iter), function_sim=sim,
                          hidden_layer_sizes=[64, 64, 64],
                          num_train_steps=20000, data_batch_size=4,
                          independent_output_dims=True)

for i in range(10):
    bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
    if NUM_DIM_X == 1:
        bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 5000}',
                    domain_l=-7, domain_u=7)
