import os.path
import pickle
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib as mpl
from plotting_hyperdata import plotting_constants

TITLE_FONT_SIZE = 26
TABLE_FONT_SIZE = 26
LEGEND_FONT_SIZE = 26
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


def calculate_density(particles: np.ndarray,
                      interval: Tuple[float, float],
                      num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    for each point [N] we estimate the density from the particle then we split the interval [min(f(x)) - 0.5, max(f(x)) + 0.5]
    and take the q-quantile of the density
    """
    assert particles.ndim == 1
    kde = gaussian_kde(particles, bw_method='scott')
    x = np.linspace(*interval, num_points)
    # Evaluate the estimated pdf on x
    density_values = kde(x)
    assert density_values.ndim == 1
    return x, density_values


def calculate_density_no_interval(particles: np.ndarray,
                                  num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    for each point [N] we estimate the density from the particle then we split the interval [min(f(x)) - 0.5, max(f(x)) + 0.5]
    and take the q-quantile of the density
    """
    assert particles.ndim == 1
    kde = gaussian_kde(particles, bw_method='scott')
    # We make an interval
    x_min = np.min(particles)
    x_max = np.max(particles)
    eps = (x_max - x_min) * 0.05 + 0.001
    x = np.linspace(x_min - eps, x_max + eps, num_points)
    # Evaluate the estimated pdf on x
    density_values = kde(x)
    assert density_values.ndim == 1
    return x, density_values


def calculate_points_with_density(x: np.ndarray,
                                  density_values: np.ndarray,
                                  q: float = 0.2):
    # Sort PDF values and associated x points in descending order of density
    sorted_indices = np.argsort(-density_values)
    sorted_density_values = density_values[sorted_indices]
    sorted_x = x[sorted_indices]

    # Calculate the cumulative sum of the PDF, normalized by the sum to get a CDF-like result
    cumulative_pdf = np.cumsum(sorted_density_values) / np.sum(sorted_density_values)

    # Find the threshold index where we cross the desired cumulative probability
    threshold_index = np.where(cumulative_pdf >= 1 - q)[0][0]

    # Select x values with the largest densities within the specified probability
    selected_x = sorted_x[:threshold_index + 1]

    selected_mask = np.isin(x, selected_x)
    return selected_mask


def create_intervals_of_mask(mask: np.array):
    # Find the start positions as locations where the value is True and the previous value is False or
    # if it's the first element of the array
    arr_shifted_left = np.roll(mask, -1)
    arr_shifted_right = np.roll(mask, 1)

    # Find start positions (False to True transition)
    start_positions = np.where((mask == True) & (arr_shifted_right == False))[0]

    # Find end positions (True to False transition). We subtract one to get the last `True` index.
    # We also handle the case where the last element is `True`.
    end_positions = np.where((mask == True) & (arr_shifted_left == False))[0]
    if mask[-1]:
        end_positions = np.append(end_positions, len(mask) - 1)

    # Combine the start and end positions into a list of tuples
    consecutive_true_ranges = list(zip(start_positions, end_positions))
    return consecutive_true_ranges


def create_intervals_of_density(particles: np.ndarray,
                                interval: Tuple[float, float],
                                num_points: int = 100,
                                q: float = 0.2) -> List[Tuple[float, float]]:
    x, density_values = calculate_density(particles, interval=interval, num_points=num_points)
    selected_mask = calculate_points_with_density(x, density_values, q=q)
    consecutive_true_ranges = create_intervals_of_mask(selected_mask)
    # Now we create consecutive true ranges:
    intervals = []
    for index_range in consecutive_true_ranges:
        start, end = index_range
        intervals.append((x[start], x[end]))
    return intervals


def prepare_plot_multimodal(ax,
                            xs,
                            ys_samples,
                            true_function,
                            train_points,
                            num_points: int = 1000,
                            q: float = 0.2,
                            plotting_eps: float = 1.0,
                            add_legend=False,
                            title: str | None = None,
                            add_x_label: bool = True,
                            add_y_label: bool = True):
    num_xs = len(xs)
    max_y = np.max(ys_samples)
    min_y = np.min(ys_samples)
    plotting_eps = plotting_eps

    # We now calculate intervals
    xs_intervals = []
    for samples_idx in range(num_xs):
        x_intervals = create_intervals_of_density(particles=ys_samples[:, samples_idx],
                                                  interval=(min_y - plotting_eps, max_y + plotting_eps),
                                                  num_points=num_points,
                                                  q=q
                                                  )
        xs_intervals.append(x_intervals)

    dx = (xs[1] - xs[0]) / 2
    # Plot high pdf regions
    for idx1, (x, ivals) in enumerate(zip(xs, xs_intervals)):
        for idx2, (y_lower, y_upper) in enumerate(ivals):
            if idx1 == idx2 == 0:
                ax.fill_between([x - dx, x + dx], [y_lower, y_lower], [y_upper, y_upper],
                                alpha=plotting_constants.CONFIDENCE_ALPHA,
                                color=plotting_constants.MEAN_FUNCTION_COLOR,
                                edgecolor=None,
                                label='High confidence region',
                                zorder=0)
            else:
                ax.fill_between([x - dx, x + dx], [y_lower, y_lower], [y_upper, y_upper],
                                alpha=plotting_constants.CONFIDENCE_ALPHA,
                                color=plotting_constants.MEAN_FUNCTION_COLOR,
                                edgecolor=None, zorder=0)
    # Plot samples:
    for i in range(ys_samples.shape[0]):
        if i == 0:
            ax.plot(xs, ys_samples[i],
                    color=plotting_constants.SAMPLES_COLOR,
                    alpha=plotting_constants.SAMPLES_ALPHA,
                    linestyle=plotting_constants.SAMPLES_LINE_STYLE,
                    linewidth=plotting_constants.SAMPLES_LINE_WIDTH,
                    label='Particles', zorder=1)
        else:
            ax.plot(xs, ys_samples[i],
                    color=plotting_constants.SAMPLES_COLOR,
                    alpha=plotting_constants.SAMPLES_ALPHA,
                    linestyle=plotting_constants.SAMPLES_LINE_STYLE,
                    linewidth=plotting_constants.SAMPLES_LINE_WIDTH,
                    zorder=1)

    # Plot true function
    ax.plot(true_function[0][:, 0], true_function[1], label='True function',
            color=plotting_constants.TRUE_FUNCTION_COLOR, zorder=2,
            linewidth=plotting_constants.TRUE_FUNCTION_LINE_WIDTH,
            linestyle=plotting_constants.TRUE_FUNCTION_LINE_STYLE)

    # Plot observations
    ax.scatter(train_points[0], train_points[1],
               label='Observations',
               color=plotting_constants.OBSERVATIONS_COLOR,
               s=OBSERVATION_SIZE,
               marker='x', zorder=3,
               linewidths=plotting_constants.OBSERVATIONS_LINE_WIDTH)
    if add_legend:
        ax.legend(fontsize=LEGEND_FONT_SIZE)

    if add_x_label:
        ax.set_xlabel(r'$\vx$', fontsize=LABEL_FONT_SIZE)
    if add_y_label:
        ax.set_ylabel(r'$\vf(\vx)$', fontsize=LABEL_FONT_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_ylim((-4, 4))


def prepare_prior_plot(ax,
                       xs: np.ndarray,
                       fs: np.ndarray,
                       title: str | None = None,
                       add_x_label: bool = True,
                       add_y_label: bool = True
                       ):
    assert xs.ndim == 2 and fs.ndim == 3
    num_samples = fs.shape[0]
    for i in range(num_samples):
        ax.plot(xs[:, 0], fs[i, :, 0],
                color=plotting_constants.SAMPLES_COLOR,
                alpha=plotting_constants.SAMPLES_ALPHA,
                linestyle=plotting_constants.SAMPLES_LINE_STYLE,
                linewidth=plotting_constants.SAMPLES_LINE_WIDTH, )

    if add_x_label:
        ax.set_xlabel(r'$\vx$', fontsize=LABEL_FONT_SIZE)
    if add_y_label:
        ax.set_ylabel(r'$\vf(\vx)$', fontsize=LABEL_FONT_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)


def one_row_plot():
    path_to_data_folder = 'results/data'
    filenames = ['gp.pkl', 'kde.pkl', 'nu_method.pkl', 'ssge.pkl']

    names = ['GP', 'KDE', 'Nu-Method', 'SSGE']

    # We join the folder path with models
    filenames = [os.path.join(path_to_data_folder, name) for name in filenames]

    data = []

    for name in filenames:
        # Open the file in read-binary mode ('rb') and load the data with pickle.load()
        with open(name, 'rb') as file:
            data.append(pickle.load(file))

    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    # Load and plot the prior samples
    filename_prior_data = 'results/data/bimodal_function_prior.pkl'
    with open(filename_prior_data, 'rb') as handle:
        prior_data = pickle.load(handle)

    prepare_prior_plot(axs[0],
                       prior_data['xs'],
                       prior_data['fs'],
                       title='Sampled Prior Functions')

    # Use the loaded data
    for idx, datum in enumerate(data):
        method = names[idx]
        datum = datum['Dimension_0']

        train_points = datum['Train points']
        true_function = datum['True function']
        sample_predictions = datum['Sample predictions']

        xs = np.array(sample_predictions[0][:, 0])
        ys_samples = np.array(sample_predictions[1])

        add_legend = True if idx == 0 else False
        prepare_plot_multimodal(axs[idx + 1], xs, ys_samples, true_function, train_points, add_legend=add_legend,
                                title=method)

    plt.savefig('multimodal_example.pdf')
    plt.show()


def two_rows_plot_with_table():
    path_to_data_folder = 'results/data'
    filenames = ['gp.pkl', 'kde.pkl', 'nu_method.pkl', 'ssge.pkl']

    # names = ['GP', 'KDE', 'Nu-Method', 'SSGE']
    names = [plotting_constants.METHODS[i] for i in [7, 8, 9, 10]]

    # We join the folder path with models
    filenames = [os.path.join(path_to_data_folder, name) for name in filenames]

    data = []

    for name in filenames:
        # Open the file in read-binary mode ('rb') and load the data with pickle.load()
        with open(name, 'rb') as file:
            data.append(pickle.load(file))

    # Load and plot the prior samples
    filename_prior_data = 'results/data/bimodal_function_prior.pkl'
    with open(filename_prior_data, 'rb') as handle:
        prior_data = pickle.load(handle)

    # Create a figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharex=True, sharey=True)

    # First row: two subplots with a dummy subplot to center them
    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    ax2 = axes[0, 2]

    # Second row: three subplots
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    ax5 = axes[1, 2]

    ########################### TOP ROW ###########################
    prepare_prior_plot(ax0,
                       prior_data['xs'],
                       prior_data['fs'],
                       title='Sampled Prior Functions',
                       add_x_label=False)

    idx = 2
    method = names[idx]
    datum = data[idx]['Dimension_0']

    train_points = datum['Train points']
    true_function = datum['True function']
    sample_predictions = datum['Sample predictions']

    xs = np.array(sample_predictions[0][:, 0])
    ys_samples = np.array(sample_predictions[1])

    prepare_plot_multimodal(ax1, xs, ys_samples, true_function, train_points,
                            add_legend=False,
                            title=method,
                            add_x_label=False,
                            add_y_label=False)

    # Make a plot of speed
    methods_data = {
        'Score Estimator': ['Nu-Method', 'GP', 'SSGE', 'KDE'],
        'Average Gradient \n Update Time (ms)': [1.19, 0.54, 0.75, 0.33]
    }
    methods_df = pd.DataFrame(methods_data)
    ax2.axis('off')
    # Create table
    the_table = ax2.table(cellText=methods_df.values, colLabels=methods_df.columns, loc='center')
    # Scale the table
    cellDict = the_table.get_celld()
    for i in range(0, len(methods_df.columns)):
        for j in range(0, len(methods_df) + 1):
            cell = cellDict[(j, i)]
            if j == 0:
                cell.set_height(.2)
            else:
                cell.set_height(.09)
            cell.set_fontsize(TABLE_FONT_SIZE)  # Change the fontsize as needed
            cell.set_text_props(horizontalalignment='center')

    # Modify table lines to look more like booktabs style
    for (i, j), cell in the_table.get_celld().items():
        if (i == 0 or i == len(methods_df)) and j == -1:
            # Set the top and bottom lines thicker (like \toprule and \bottomrule)
            cell.visible_edges = "T" if i == 0 else "B"
            cell.set_linewidth(3)
        elif i == 0:
            # For the line below the header, make the line a bit thicker (like \midrule)
            cell.visible_edges = "T"
            cell.set_linewidth(3)
        elif i == 1:
            # For the line below the header, make the line a bit thicker (like \midrule)
            cell.visible_edges = "T"
            cell.set_linewidth(1.5)
        elif i == 4:
            # For the line below the header, make the line a bit thicker (like \midrule)
            cell.visible_edges = "B"
            cell.set_linewidth(3)
        else:
            # Remove all other interior lines
            cell.visible_edges = ""

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(TABLE_FONT_SIZE)  # This sets the default fontsize

    the_table.scale(1, 1.5)

    ########################### BOTTOM ROW ###########################
    indices = [0, 1, 3]
    axs_bottom = [ax3, ax4, ax5]
    for counter, idx in enumerate(indices):
        method = names[idx]
        datum = data[idx]['Dimension_0']

        train_points = datum['Train points']
        true_function = datum['True function']
        sample_predictions = datum['Sample predictions']

        xs = np.array(sample_predictions[0][:, 0])
        ys_samples = np.array(sample_predictions[1])

        prepare_plot_multimodal(axs_bottom[counter], xs, ys_samples, true_function, train_points, add_legend=False,
                                title=method,
                                add_y_label=True if counter == 0 else False
                                )

    # Add the legend with handles and labels, specify the location and number of columns
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               ncols=4,
               loc='upper center',
               fontsize=LEGEND_FONT_SIZE,
               frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('multimodal_example.pdf')
    plt.show()


if __name__ == '__main__':
    two_rows_plot_with_table()
