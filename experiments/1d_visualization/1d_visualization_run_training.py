from sim_transfer.sims.simulators import SinusoidsSim, QuadraticSim, LinearSim, ShiftedSinusoidsSim
from sim_transfer.models import BNN_FSVGD_SimPrior, BNN_SVGD, BNN_FSVGD
from matplotlib import pyplot as plt

import pickle
import os
import jax
import jax.numpy as jnp


# determine plot_dict_dir
plot_dict_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plot_dicts')
os.makedirs(plot_dict_dir, exist_ok=True)


def _key_iter(data_seed: int = 24359):
    key = jax.random.PRNGKey(data_seed)
    while True:
        key, new_key = jax.random.split(key)
        yield new_key


def main(sim_type: str = 'SinusoidsSim', model: str = 'BNN_FSVGD_SimPrior_gp', num_train_points: int = 1,
         plot_post_samples: bool = True, fun_seed: int = 24359):
    key_iter = _key_iter()

    if sim_type == 'QuadraticSim':
        sim = QuadraticSim()
    elif sim_type == 'LinearSim':
        sim = LinearSim()
    elif sim_type == 'SinusoidsSim':
        sim = SinusoidsSim(output_size=1)
    else:
        raise NotImplementedError

    x_plot = jnp.linspace(sim.domain.l, sim.domain.u, 100).reshape(-1, 1)

    # """ plot samples from the simulation env """
    # f_sim = sim.sample_function_vals(x_plot, num_samples=10, rng_key=jax.random.PRNGKey(234234))
    # for i in range(f_sim.shape[0]):
    #     plt.plot(x_plot, f_sim[i])
    # plt.show()

    """ generate data """
    fun = sim.sample_function(rng_key=jax.random.PRNGKey(291))  # 764
    x_train = jax.random.uniform(key=next(key_iter), shape=(50,),
                                 minval=sim.domain.l, maxval=sim.domain.u).reshape(-1, 1)
    x_train = x_train[:num_train_points]
    y_train = fun(x_train)
    x_test = jnp.linspace(sim.domain.l, sim.domain.u, 100).reshape(-1, 1)
    y_test = fun(x_test)

    """ fit the model """
    common_kwargs = {
        'input_size': 1,
        'output_size': 1,
        'rng_key': next(key_iter),
        'hidden_layer_sizes': [64, 64, 64],
        'data_batch_size': 4,
        'num_particles': 20,
        'likelihood_std': 0.05,
        'normalization_stats': sim.normalization_stats,
    }
    if model == 'BNN_SVGD':
        bnn = BNN_SVGD(**common_kwargs, bandwidth_svgd=10., num_train_steps=2)
    elif model == 'BNN_FSVGD':
        bnn = BNN_FSVGD(**common_kwargs, domain=sim.domain, bandwidth_svgd=0.5, num_measurement_points=8)
    elif model == 'BNN_FSVGD_SimPrior_gp':
        bnn = BNN_FSVGD_SimPrior(**common_kwargs, domain=sim.domain, function_sim=sim,
                                 num_train_steps=20000, num_f_samples=256, num_measurement_points=8,
                                 bandwidth_svgd=1., score_estimator='gp')
    elif model == 'BNN_FSVGD_SimPrior_kde':
        bnn = BNN_FSVGD_SimPrior(**common_kwargs, domain=sim.domain, function_sim=sim,
                                 num_train_steps=40000, num_f_samples=256, num_measurement_points=16,
                                 bandwidth_svgd=1., score_estimator='kde')
    elif model == 'BNN_FSVGD_SimPrior_nu-method':
        bnn = BNN_FSVGD_SimPrior(**common_kwargs, domain=sim.domain, function_sim=sim,
                                 num_train_steps=20000, num_f_samples=256, num_measurement_points=16,
                                 bandwidth_svgd=1., score_estimator='nu-method', bandwidth_score_estim=1.0)

    else:
        raise NotImplementedError

    bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test)

    """ make predictions and save the plot """
    x_plot = jnp.linspace(sim.domain.l, sim.domain.u, 200).reshape((-1, 1))

    # make predictions
    pred_mean, pred_std = bnn.predict(x_plot)
    y_post_samples = bnn.predict_post_samples(x_plot)

    # get true function value
    true_fun = fun(x_plot)
    typical_fun = sim._typical_f(x_plot)

    plot_dict = {
        'model': model,
        'plot_data': {
            'x_train': x_train,
            'y_train': y_train,
            'x_plot': x_plot,
            'true_fun': true_fun,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'y_post_samples': y_post_samples,
        }
    }
    dump_path = os.path.join(plot_dict_dir, f'{sim_type}_{model}_{num_train_points}.pkl')

    with open(dump_path, 'wb') as f:
        pickle.dump(plot_dict, f)
    print(f'Plot dict saved to {dump_path}')

    # draw the plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1 * 4, 4))
    if bnn.output_size == 1:
        ax = [ax]
    for i in range(1):
        ax[i].scatter(x_train.flatten(), y_train[:, i], label='train points')
        ax[i].plot(x_plot, fun(x_plot)[:, i], label='true fun')
        ax[i].plot(x_plot, typical_fun, label='typical fun')
        ax[i].plot(x_plot.flatten(), pred_mean[:, i], label='pred mean')
        ax[i].fill_between(x_plot.flatten(), pred_mean[:, i] - 2 * pred_std[:, i],
                           pred_mean[:, i] + 2 * pred_std[:, i], alpha=0.3)

        if plot_post_samples:
            y_post_samples = bnn.predict_post_samples(x_plot)
            for y in y_post_samples:
                ax[i].plot(x_plot, y[:, i], linewidth=0.2, color='green')

        ax[i].legend()
    fig.suptitle(model)
    fig.show()


if __name__ == '__main__':
    for num_train_points in [2]: #, 3, 5]:
        for model in [
            'BNN_SVGD',
            'BNN_FSVGD',
            'BNN_FSVGD_SimPrior_gp',
            'BNN_FSVGD_SimPrior_nu-method',
            'BNN_FSVGD_SimPrior_kde'
        ]:
            main(model=model, num_train_points=num_train_points)