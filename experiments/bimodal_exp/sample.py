import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import cond

import wandb
from sim_transfer.models.bnn_fsvgd_sim_prior import BNN_FSVGD_SimPrior
from sim_transfer.sims import LinearBimodalSim

if __name__ == '__main__':
    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key


    key_iter = key_iter()
    NUM_DIM_X = 1
    NUM_DIM_Y = 1

    sim = LinearBimodalSim()


    def fun(x):
        def positive(x):
            return x

        def negative(x):
            return -0.55 * x

        return cond(x.reshape() > 0, positive, negative, x)


    v_fun = vmap(fun)
    domain = sim.domain

    num_train_points = 3
    score_estimator = 'gp'

    x_train = jax.random.uniform(key=next(key_iter), shape=(num_train_points,),
                                 minval=domain.l, maxval=jnp.array([0.0])).reshape(-1, 1)
    y_train = v_fun(x_train)

    x_test = jnp.linspace(domain.l, domain.u, 100).reshape(-1, 1)
    y_test = v_fun(x_test)

    bnn_config = dict(domain=domain, rng_key=next(key_iter), function_sim=sim,
                      hidden_layer_sizes=[64, 64, 64], num_train_steps=4000, data_batch_size=4,
                      num_particles=20, num_f_samples=256, num_measurement_points=16,
                      bandwidth_svgd=1., bandwidth_score_estim=1.0, ssge_kernel_type='IMQ',
                      normalization_stats=sim.normalization_stats, likelihood_std=0.05,
                      score_estimator=score_estimator)

    bnn = BNN_FSVGD_SimPrior(NUM_DIM_X, NUM_DIM_Y, **bnn_config)

    wandb.init(
        project="Linear Bimodal",
        group='Testing',
        config=bnn_config,
    )

    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000, log_to_wandb=True)
        if NUM_DIM_X == 1:
            bnn.plot_1d(x_train, y_train, true_fun=v_fun,
                        title=f'FSVGD SimPrior {score_estimator}, iter {(i + 1) * 2000}',
                        domain_l=domain.l[0], domain_u=domain.u[0], log_to_wandb=True)
