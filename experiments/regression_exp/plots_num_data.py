import argparse
import math
from typing import Tuple

import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from experiments.util import collect_exp_results, ucb, lcb, median, count
from plotting_hyperdata import plotting_constants

plt.locator_params(nbins=4)

TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 14
LABEL_FONT_SIZE = 18
YLABEL_FONT_SIZE = 18
XLABEL_FONT_SIZE = 18
TICKS_SIZE = 14
LINE_WIDTH = 3

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE


class FirstNonZeroFormatter(ticker.Formatter):
    def __call__(self, x, pos=None):
        # Handling very small or zero values to avoid log errors
        if x <= 0:
            return '0'

        # Calculate the number of decimal places needed
        # This will be negative for numbers larger than 1
        # and positive for numbers smaller than 1.
        num_decimals = np.ceil(-np.log10(x)).astype(int)

        # Format string for precision
        format_str = '{{:.{}f}}'.format(max(num_decimals, 0))

        # Format the tick label
        formatted_label = format_str.format(x)

        # If the resulting formatted label has a period at the end, remove it
        # This can happen when x is an exact integer
        if formatted_label.endswith('.'):
            formatted_label = formatted_label[:-1]

        return formatted_label


def different_method_plot(df_agg: pd.DataFrame, metric: str = 'nll', display: bool = True,
                          filter_std_higher_than: float = math.inf) \
        -> Tuple[pd.DataFrame, plt.Figure]:
    models = set(df_agg['model'])

    best_rows = []
    for model in models:
        df_agg_model = df_agg.loc[df_agg['model'] == model]
        df_agg_model = df_agg_model[df_agg_model[(metric, 'std')] < filter_std_higher_than]
        df_agg_model.sort_values(by=(metric, 'mean'), ascending=True, inplace=True)
        best_rows.append(df_agg_model.iloc[0])

    best_rows_df = pd.DataFrame(best_rows)

    fig, ax = plt.subplots()
    x_pos = 2 * np.arange(len(models))
    ax.bar(x_pos, best_rows_df[(metric, 'mean')], color='#ADD8E6')
    ax.errorbar(x_pos, best_rows_df[(metric, 'mean')], yerr=best_rows_df[(metric, 'std')],
                linestyle='', capsize=4., color='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(best_rows_df['model'], rotation=-20)

    if display:
        plt.show()
        print(best_rows_df[[('model', ''), ('nll', 'mean'), ('nll', 'std'),
                            ('rmse', 'mean'), ('rmse', 'std'), ('dirname', '')]].to_string())

    return best_rows_df, fig


def main(args, drop_nan=False, fig=None, show_legend=True, log_scale=False):
    df_full, param_names = collect_exp_results(exp_name=args.exp_name)
    df_full = df_full[df_full['data_source'] == args.data_source]
    df_full = df_full[df_full['num_samples_train'] >= 10]

    for col in ['bandwidth_kde', 'bandwidth_ssge', 'bandwidth_score_estim']:
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna(value='auto')

    if drop_nan:
        result_is_nan = df_full['nll'].isna()
        df_full_dropped = df_full[result_is_nan]
        print(f'Dropped results due to NaNs: {df_full_dropped.shape[0]}')
        df_full = df_full[~result_is_nan]
    else:
        # replace nan with bad values
        for col, noise_std in [('nll', 1000.), ('rmse', 5.)]:
            series = df_full[col]
            max_value = series.max()
            is_nan = series.isna()
            noise = np.random.normal(0, noise_std, size=is_nan.sum())  # Adjust mean and stddev as needed
            series[is_nan] = max_value + noise
            df_full[col] = series

    # group over everything except seeds and aggregate over the seeds
    groupby_names = list(set(param_names) - {'model_seed', 'data_seed'})

    # rmove the likelihood_std column since it's a constant list which is not hashable
    groupby_names.remove('likelihood_std')
    # groupby_names.remove('added_gp_outputscale')
    # df_full['added_gp_outputscale'] = df_full['added_gp_outputscale'].apply(lambda x: x[0])

    # replace all the nans in hyperparameter columns with 'N/A'
    for column in groupby_names:
        df_full[column] = df_full[column].fillna('N/A')

    # first take mean over the data seeds
    df_mean = df_full.groupby(by=groupby_names + ['model_seed'], axis=0).mean(numeric_only=True)
    df_mean.reset_index(drop=False, inplace=True)

    # then compute the stats over the model seeds
    df_agg = df_mean.groupby(by=groupby_names, axis=0).aggregate(['mean', 'std', ucb, lcb, median, count], axis=0)
    df_agg.reset_index(drop=False, inplace=True)
    # df_agg.sort_values(by=[('nll', 'mean')], ascending=True, inplace=True)

    # filter all the rows where the count is less than 3
    df_agg = df_agg[df_agg['rmse']['count'] >= 3]

    available_models = sorted(list(set(df_agg['model'])))
    print(available_models)
    if fig is None:
        fig, axs = plt.subplots(2, 1)
    else:
        axs = fig.subplots(2, 1, sharex=True)
    for idx, metric in enumerate(['nll', 'rmse']):
        for model in available_models:
            df_model = df_agg[df_agg['model'] == model].sort_values(by=[('num_samples_train', '')], ascending=True)
            nice_name = plotting_constants.plot_num_data_name_transfer[model]
            if args.quantile_cis:
                axs[idx].plot(df_model[('num_samples_train', '')], df_model[(metric, 'median')],
                              label=nice_name,
                              color=plotting_constants.COLORS[nice_name],
                              linestyle=plotting_constants.LINE_STYLES[nice_name],
                              linewidth=LINE_WIDTH)
                lower_ci = df_model[(metric, 'lcb')]
                upper_ci = df_model[(metric, 'ucb')]
            else:
                axs[idx].plot(df_model[('num_samples_train', '')], df_model[(metric, 'mean')],
                              label=nice_name,
                              color=plotting_constants.COLORS[nice_name],
                              linestyle=plotting_constants.LINE_STYLES[nice_name],
                              linewidth=LINE_WIDTH)
                CI_width = 2 / np.sqrt(df_model[(metric, 'count')])
                lower_ci = df_model[(metric, 'mean')] - CI_width * df_model[(metric, 'std')]
                upper_ci = df_model[(metric, 'mean')] + CI_width * df_model[(metric, 'std')]
            axs[idx].fill_between(df_model[('num_samples_train', '')], lower_ci, upper_ci, alpha=0.3,
                                  color=plotting_constants.COLORS[nice_name])
        if idx == 0:
            axs[idx].set_title(plotting_constants.plot_num_data_data_source_transfer[args.data_source],
                               fontsize=TITLE_FONT_SIZE)
        if args.data_source == 'racecar':
            axs[idx].set_ylabel(plotting_constants.plot_num_data_metrics_transfer[metric],
                                fontsize=YLABEL_FONT_SIZE)
        if idx == 1 and log_scale:
            axs[idx].set_yscale('log')
            axs[idx].yaxis.set_major_formatter(FirstNonZeroFormatter())
            axs[idx].yaxis.set_minor_formatter(FirstNonZeroFormatter())
            axs[idx].yaxis.set_minor_locator(plt.MaxNLocator(6))
            axs[idx].set_xlabel('Number of iterations', fontsize=XLABEL_FONT_SIZE)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if show_legend:
        fig.legend(by_label.values(), by_label.keys(), ncols=4, loc='lower center',
                   bbox_to_anchor=(0.5, 0), fontsize=10)

    # plt.show()
    print('Models:', set(df_agg['model']))


if __name__ == '__main__':
    figure = plt.figure(figsize=(10, 7))
    subfigs = figure.subfigures(1, 2, wspace=-0.05)

    parser = argparse.ArgumentParser(description='Inspect results of a regression experiment.')
    parser.add_argument('--exp_name', type=str, default='jan31')
    parser.add_argument('--quantile_cis', type=int, default=1)
    parser.add_argument('--data_source', type=str, default='racecar_hf')
    args = parser.parse_args()

    main(args, fig=subfigs[0], show_legend=False, log_scale=True)

    parser = argparse.ArgumentParser(description='Inspect results of a regression experiment.')
    parser.add_argument('--exp_name', type=str, default='jan31')
    parser.add_argument('--quantile_cis', type=int, default=1)
    parser.add_argument('--data_source', type=str, default='pendulum_hf')
    args = parser.parse_args()

    main(args, fig=subfigs[1], show_legend=False, log_scale=True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(),
                  ncols=4,
                  loc='upper center',
                  fontsize=LEGEND_FONT_SIZE,
                  frameon=False)
    figure.tight_layout(rect=[0.15, -0.1, 1, 0.95])
    plt.savefig('regression_exp.pdf')
    plt.show()
