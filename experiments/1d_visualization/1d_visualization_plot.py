from matplotlib import pyplot as plt
from sim_transfer.sims.simulators import SinusoidsSim

import os
import pickle
import jax

PLOT_POST_SAMPLES = True

PLOTS_1D_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DICT_DIR = os.path.join(PLOTS_1D_DIR, 'plot_dicts')
PLOT_DICT_PATHS = [
    ('BNN_SVGD', 'SinusoidsSim_BNN_SVGD_2.pkl'),
    ('BNN_FSVGD', 'SinusoidsSim_BNN_FSVGD_2.pkl'),
    ('BNN_FSVGD_SimPrior_gp', 'SinusoidsSim_BNN_FSVGD_SimPrior_gp_2.pkl'),
    ('BNN_FSVGD_SimPrior_nu-method', 'SinusoidsSim_BNN_FSVGD_SimPrior_nu-method_2.pkl'),
    ('BNN_FSVGD_SimPrior_kde', 'SinusoidsSim_BNN_FSVGD_SimPrior_kde_2.pkl'),
]
PLOT_DICT_PATHS = map(lambda x: (x[0], os.path.join(PLOT_DICT_DIR, x[1])), PLOT_DICT_PATHS)

PLOT_MODELS = ['BNN_SVGD', 'BNN_FSVGD', 'BNN_FSVGD_SimPrior_gp']


# draw the plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(3 * 4, 6))

sim = SinusoidsSim(output_size=1)

for i, (model, load_path) in enumerate(PLOT_DICT_PATHS):
    with open(load_path, 'rb') as f:
        plot_dict = pickle.load(f)
    print(f'Plot dict loaded from {load_path}')
    plot_data = plot_dict['plot_data']

    if i == 0:
        """ plot samples from the simulation env """
        f_sim = sim.sample_function_vals(plot_data['x_plot'], num_samples=10, rng_key=jax.random.PRNGKey(234234))
        for j in range(f_sim.shape[0]):
            axes[0][0].plot(plot_data['x_plot'], f_sim[j])
        axes[0][0].set_title('sampled functions from sim prior')
        axes[0][0].set_ylim((-14, 14))


    ax = axes[(i+1)//3][(i+1)%3]
    if PLOT_POST_SAMPLES:
        for k, y in enumerate(plot_data['y_post_samples']):
            ax.plot(plot_data['x_plot'], y[:, i], linewidth=0.2, color='tab:green', alpha=0.5,
                    label='BNN particles' if k == 0 else None)

    ax.scatter(plot_data['x_train'].flatten(), plot_data['y_train'][:, i], 100, label='train points', marker='x',
               linewidths=2.5, color='tab:blue')
    ax.plot(plot_data['x_plot'], plot_data['true_fun'], label='true fun')
    ax.plot(plot_data['x_plot'].flatten(), plot_data['pred_mean'][:, i], label='pred mean')
    ax.fill_between(plot_data['x_plot'].flatten(), plot_data['pred_mean'][:, i] - 2 * plot_data['pred_std'][:, i],
                       plot_data['pred_mean'][:, i] + 2 * plot_data['pred_std'][:, i], alpha=0.2,
                       label='95 % CI', color='tab:orange')

    if i == 4:
        ax.legend()
    ax.set_title(model)
    ax.set_ylim((-14, 14))
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(PLOTS_1D_DIR, '1d_visualization.pdf'))
