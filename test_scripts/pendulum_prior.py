import jax
import jax.numpy as jnp
from jax import vmap

import wandb
from sim_transfer.models.bnn_fsvgd_sim_prior import BNN_FSVGD_Sim_Prior
from sim_transfer.sims import PendulumSim, GaussianProcessSim
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
MODEL_PRIOR = True

NUM_DIM_X = x_dim + u_dim
NUM_DIM_Y = x_dim

num_train_points = 10
d_l, d_u = -1, 1

# Here we define the true model
true_pendulum_params = PendulumParams(m=jnp.array(1.0), l=jnp.array(1.0), g=jnp.array(0.81), nu=jnp.array(0.1))
pendulum_model = Pendulum(h=time_step)

# Here we define prior
if MODEL_PRIOR:
    lower_b_params = PendulumParams(m=jnp.array(0.99), l=jnp.array(0.99), g=jnp.array(0.80), nu=jnp.array(0.09))
    upper_b_params = PendulumParams(m=jnp.array(1.01), l=jnp.array(1.01), g=jnp.array(0.82), nu=jnp.array(0.11))
    sim = PendulumSim(h=time_step, lower_bound=lower_b_params, upper_bound=upper_b_params)
else:
    sim = GaussianProcessSim(input_size=3, output_size=2)


def fun(z):
    assert z.shape == (NUM_DIM_X,)
    x, u = z[:x_dim], z[x_dim:]
    return pendulum_model.ode(x, u, true_pendulum_params)


noise_std = 0.1
x_train = jax.random.uniform(next(key_iter), shape=(num_train_points, NUM_DIM_X), minval=d_l, maxval=d_u)
y_train = vmap(fun)(x_train) + noise_std * jax.random.normal(next(key_iter), shape=(x_train.shape[0], NUM_DIM_Y))

num_test_points = 100
x_test = jax.random.uniform(next(key_iter), shape=(num_test_points, NUM_DIM_X), minval=d_l, maxval=d_u)
y_test = vmap(fun)(x_test) + noise_std * jax.random.normal(next(key_iter), shape=(x_test.shape[0], NUM_DIM_Y))
domain_l, domain_u = d_l * jnp.ones(shape=(NUM_DIM_X)), d_u * jnp.ones(shape=(NUM_DIM_X))

bnn = BNN_FSVGD_Sim_Prior(NUM_DIM_X, NUM_DIM_Y, domain_l, domain_u, rng_key=next(key_iter), function_sim=sim,
                          hidden_layer_sizes=[64, 64, 64], num_train_steps=20000, data_batch_size=10,
                          num_measurement_points=10, independent_output_dims=True, log_wandb=True,
                          likelihood_std=noise_std, num_particles=10, num_f_samples=20, bandwidth_ssge=2,
                          bandwidth_svgd=1e-1)

wandb.init(
    project="Model vs GP prior",
    group='Model prior' if MODEL_PRIOR else 'GP prior',
)

for i in range(10):
    bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
