from experiments.util import collect_exp_results, ucb, lcb, median, count, BASE_DIR
import numpy as np
from matplotlib import pyplot as plt
import os


PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

QUANTILE_BASED_CI = True
Y_AX_LABEL_MAP = {
    'x_diff': 'abs gap x',
    'f_diff': 'abs gap f',
}

PLOT_DICTS = [
    {'title': 'Some random plot',
     'file_name': 'tryout_plot',
     'metrics': ['x_diff', 'f_diff'],
     'logscale': True,
     'plot_dirs': [
         ('Random Search', 'test_may23/quadratic_random_search_100/3934868503092991762'),
         ('Hill Climbing', 'test_may23/quadratic_hill_search_100/6741021242040967667'),
     ],
     },
]


for plot_dict in PLOT_DICTS:
    n_metrics = len(plot_dict['metrics'])
    fig, axes = plt.subplots(ncols=n_metrics, figsize=(4 * n_metrics, 4))
    methods = [method for method, dir in plot_dict['plot_dirs']]
    for k, (method, result_dir) in enumerate(plot_dict['plot_dirs']):

        df_full, param_names = collect_exp_results(exp_name=result_dir, dir_tree_depth=1)

        # remove columns that only contain nans
        nan_col_mask = df_full.isna().apply(lambda col: np.all(col), axis=0)
        nan_cols = list(df_full.columns[nan_col_mask])
        df_full = df_full.drop(nan_cols, axis=1)

        # group over everything except seeds and aggregate over the seeds
        groupby_names = list(set(param_names) - set(nan_cols) - {'seed', 'model_seed'})
        df_agg = df_full.groupby(by=groupby_names, axis=0).aggregate(['mean', 'std', ucb, lcb, median, count], axis=0)
        df_agg.reset_index(drop=False, inplace=True)

        for i, metric in enumerate(plot_dict['metrics']):
            for j in range(1):
                row = df_agg.iloc[j]
                num_seeds = row[(metric, 'count')]
                ci_factor = 1 / np.sqrt(num_seeds)

                if QUANTILE_BASED_CI:
                    metric_median = row[(metric, 'median')]
                    axes[i].scatter(k, metric_median, label=f'{method}_{j}')
                    lower_err = - (row[(metric, 'lcb')] - metric_median) * ci_factor
                    upper_err = (row[(metric, 'ucb')] - metric_median) * ci_factor
                    axes[i].errorbar(k, metric_median, yerr=np.array([[lower_err, upper_err]]).T,
                                     capsize=5)
                else:
                    metric_mean = row[(metric, 'mean')]
                    metric_std = row[(metric, 'std')]
                    axes[i].scatter(k, metric_mean, label=f'{method}_{j}')
                    axes[i].errorbar(k, metric_mean, yerr=2 * metric_std * ci_factor)

                if i == 0:
                    print(f'{method}_{j}', row['exp_result_folder'][0])
            if k == 0:
                axes[i].set_ylabel(metric)
            axes[i].set_xticks(np.arange(len(methods)))
            axes[i].set_xticklabels(methods)
            axes[i].set_xlim((-0.5, len(methods) - 0.5))

            if plot_dict['logscale']:
                axes[i].set_yscale('log')

    fig.suptitle(plot_dict['title'])
    plt.tight_layout()
    plt.legend()
    fig.show()
    fig_path = os.path.join(PLOT_DIR, plot_dict['file_name'] + '.png')
    fig.savefig(fig_path)
    fig_path = os.path.join(PLOT_DIR, plot_dict['file_name'] + '.pdf')
    fig.savefig(fig_path)
    print(f'Saved figure to {fig_path}')