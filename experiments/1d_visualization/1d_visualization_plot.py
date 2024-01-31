from matplotlib import pyplot as plt
from sim_transfer.sims.simulators import SinusoidsSim
from plotting_hyperdata import plotting_constants
import matplotlib as mpl

import os
import pickle
import jax

PLOT_POST_SAMPLES = True
LEGEND_FONT_SIZE = 26
TITLE_FONT_SIZE = 26
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE

PLOTS_1D_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DICT_DIR = os.path.join(PLOTS_1D_DIR, 'plot_dicts')
PLOT_DICT_PATHS = [
    (plotting_constants.METHODS[0], 'SinusoidsSim_BNN_SVGD_2.pkl'),
    (plotting_constants.METHODS[1], 'SinusoidsSim_BNN_FSVGD_2.pkl'),
    (plotting_constants.METHODS[2], 'SinusoidsSim_GreyBox_2.pkl'),
    (plotting_constants.METHODS[4], 'SinusoidsSim_BNN_FSVGD_SimPrior_gp_2.pkl'),
    (plotting_constants.METHODS[6], 'SinusoidsSim_BNN_FSVGD_SimPrior_nu-method_2.pkl'),
]
PLOT_DICT_PATHS = map(lambda x: (x[0], os.path.join(PLOT_DICT_DIR, x[1])), PLOT_DICT_PATHS)

PLOT_MODELS = ['BNN_SVGD', 'BNN_FSVGD', 'BNN_FSVGD_SimPrior_gp']

# draw the plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharex=True, sharey=True)

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
            axes[0][0].plot(plot_data['x_plot'], f_sim[j],
                            color=plotting_constants.SAMPLES_COLOR,
                            linestyle=plotting_constants.SAMPLES_LINE_STYLE,
                            linewidth=plotting_constants.SAMPLES_LINE_WIDTH,
                            alpha=plotting_constants.SAMPLES_ALPHA)
        axes[0][0].set_title(plotting_constants.METHODS[8], fontsize=TITLE_FONT_SIZE)
        axes[0][0].set_ylim((-14, 14))
        axes[0][0].set_ylabel(r'$\vf(\vx)$', fontsize=LABEL_FONT_SIZE)

    ax = axes[(i + 1) // 3][(i + 1) % 3]
    if PLOT_POST_SAMPLES:
        for k, y in enumerate(plot_data['y_post_samples']):
            ax.plot(plot_data['x_plot'], y[:, i],
                    linewidth=plotting_constants.SAMPLES_LINE_WIDTH,
                    linestyle=plotting_constants.SAMPLES_LINE_STYLE,
                    color=plotting_constants.SAMPLES_COLOR,
                    alpha=plotting_constants.SAMPLES_ALPHA,
                    label='Particles' if k == 0 else None,
                    zorder=1)

    ax.scatter(plot_data['x_train'].flatten(), plot_data['y_train'][:, i], s=OBSERVATION_SIZE, label='Observations',
               marker='x', linewidths=plotting_constants.OBSERVATIONS_LINE_WIDTH,
               color=plotting_constants.OBSERVATIONS_COLOR,
               zorder=4)
    ax.plot(plot_data['x_plot'], plot_data['true_fun'],
            color=plotting_constants.TRUE_FUNCTION_COLOR,
            linestyle=plotting_constants.TRUE_FUNCTION_LINE_STYLE,
            linewidth=plotting_constants.TRUE_FUNCTION_LINE_WIDTH,
            label='True Function',
            zorder=2)
    ax.plot(plot_data['x_plot'].flatten(), plot_data['pred_mean'][:, i],
            color=plotting_constants.MEAN_FUNCTION_COLOR,
            linestyle=plotting_constants.MEAN_FUNCTION_LINE_STYLE,
            linewidth=plotting_constants.MEAN_FUNCTION_LINE_WIDTH,
            label='Mean function',
            zorder=3)
    ax.fill_between(plot_data['x_plot'].flatten(), plot_data['pred_mean'][:, i] - 2 * plot_data['pred_std'][:, i],
                    plot_data['pred_mean'][:, i] + 2 * plot_data['pred_std'][:, i],
                    label='High confidence region',
                    alpha=plotting_constants.CONFIDENCE_ALPHA,
                    color=plotting_constants.MEAN_FUNCTION_COLOR,
                    zorder=0)

    ax.set_title(model, fontsize=TITLE_FONT_SIZE)
    if i == 2:
        ax.set_ylabel(r'$\vf(\vx)$', fontsize=LABEL_FONT_SIZE)
    if i >= 2:
        ax.set_xlabel(r'$\vx$', fontsize=LABEL_FONT_SIZE)
    ax.set_ylim((-14, 14))

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           ncols=5,
           loc='upper center',
           # bbox_to_anchor=(0.5, 0),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(PLOTS_1D_DIR, '1d_visualization.pdf'))
fig.show()
