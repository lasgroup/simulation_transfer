import os
import pickle
from matplotlib import pyplot as plt

PLOT_POST_SAMPLES = True

PLOT_DICT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_dicts')
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


for i, (model, load_path) in enumerate(PLOT_DICT_PATHS):
    with open(load_path, 'rb') as f:
        plot_dict = pickle.load(f)
    print(f'Plot dict loaded from {load_path}')
    plot_data = plot_dict['plot_data']

    ax = axes[i//3][i%3]
    ax.scatter(plot_data['x_train'].flatten(), plot_data['y_train'][:, i], label='train points')
    ax.plot(plot_data['x_plot'], plot_data['true_fun'], label='true fun')
    ax.plot(plot_data['x_plot'].flatten(), plot_data['pred_mean'][:, i], label='pred mean')
    ax.fill_between(plot_data['x_plot'].flatten(), plot_data['pred_mean'][:, i] - 2 * plot_data['pred_std'][:, i],
                       plot_data['pred_mean'][:, i] + 2 * plot_data['pred_std'][:, i], alpha=0.3)

    if PLOT_POST_SAMPLES:
        for y in plot_data['y_post_samples']:
            ax.plot(plot_data['x_plot'], y[:, i], linewidth=0.2, color='green')

    if i == 2:
        ax.legend()
    ax.set_title(model)
    ax.set_ylim((-14, 14))
fig.tight_layout()
fig.show()
