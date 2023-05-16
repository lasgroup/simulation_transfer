import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap
from jax.config import config

from sim_transfer.models.bnn_mmd_sim_prior import BNN_MMD_SimPrior
from sim_transfer.models.bnn_fsvgd_sim_prior import BNN_FSVGD_SimPrior
from sim_transfer.models.bnn_wasserstein_sim_prior import BNN_Wasserstein_SimPrior
from sim_transfer.sims import GaussianProcessSim, QuadraticSim
from sim_transfer.sims.domain import HypercubeDomain


# config.update("jax_enable_x64", True)


def key_iter():
    key = jax.random.PRNGKey(7644)
    while True:
        key, new_key = jax.random.split(key)
        yield new_key


key_iter = key_iter()
MODEL_PRIOR = True

NUM_DIM_X = 1
NUM_DIM_Y = 1

num_train_points = 4

# Here we define the true model


# Here we define prior
if MODEL_PRIOR:
    sim = QuadraticSim()
else:
    sim = GaussianProcessSim(input_size=1, output_size=1)


def fun(z):
    # assert z.shape == (1,)
    return (z - 2) ** 2


d_l, d_u = 0, 4

num_test_functions = 30
sample_xs = jnp.linspace(d_l, d_u, 100).reshape(-1, 1)
sample_fns = sim.sample_function_vals(sample_xs, num_test_functions, rng_key=jax.random.PRNGKey(0))
for i in range(num_test_functions):
    plt.plot(sample_xs, sample_fns[i, ...])
plt.show()

x_train = jax.random.uniform(next(key_iter), shape=(num_train_points, NUM_DIM_X), minval=d_l, maxval=d_u)
y_train = vmap(fun)(x_train) + 0.1 * jax.random.normal(next(key_iter), shape=(x_train.shape[0], NUM_DIM_Y))

num_test_points = 100
x_test = jax.random.uniform(next(key_iter), shape=(num_test_points, NUM_DIM_X), minval=d_l, maxval=d_u)
y_test = vmap(fun)(x_test) + 0.1 * jax.random.normal(next(key_iter), shape=(x_test.shape[0], NUM_DIM_Y))
domain_l, domain_u = d_l * jnp.ones(shape=(NUM_DIM_X)), d_u * jnp.ones(shape=(NUM_DIM_X))
domain = HypercubeDomain(domain_l, domain_u)

# bnn = BNN_FSVGD_Sim_Prior(NUM_DIM_X, NUM_DIM_Y, domain, rng_key=next(key_iter), function_sim=sim,
#                           hidden_layer_sizes=[64, 64, 64], num_train_steps=20000, data_batch_size=num_train_points,
#                           num_measurement_points=10, independent_output_dims=True, num_f_samples=20,
#                           bandwidth_svgd=1e-4, bandwidth_score_estim=1.)

# bnn = BNN_MMD_SimPrior(NUM_DIM_X, NUM_DIM_Y, domain, rng_key=next(key_iter), function_sim=sim,
#                        hidden_layer_sizes=[64, 64, 64], num_train_steps=20000, data_batch_size=num_train_points,
#                        num_measurement_points=10, independent_output_dims=True, num_f_samples=20, num_particles=7)

bnn = BNN_Wasserstein_SimPrior(NUM_DIM_X, NUM_DIM_Y, domain, rng_key=next(key_iter), function_sim=sim,
                       hidden_layer_sizes=[64, 64, 64], num_train_steps=20000, data_batch_size=num_train_points,
                       num_measurement_points=20, independent_output_dims=True, num_f_samples=30, num_particles=7)

for i in range(10):
    bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
    bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i + 1) * 5000}',
                domain_l=d_l, domain_u=d_u)
