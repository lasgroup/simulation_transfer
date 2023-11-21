import pandas as pd
import numpy as np
import argparse

from matplotlib import pyplot as plt
from experiments.util import collect_exp_results, ucb, lcb, median, count



def main(args):
    df_full, param_names = collect_exp_results(exp_name=args.exp_name)

    # group over everything except seeds and aggregate over the seeds
    groupby_names = list(set(param_names) - {'seed'})
    df_agg = df_full.groupby(by=groupby_names, axis=0).aggregate(['mean', 'std', ucb, lcb, median, count], axis=0)
    df_agg.reset_index(drop=False, inplace=True)

    print('Available methods:', set(df_agg['method']))

    QUANTILE_BASED_CI = True
    METHODS = ['random_search', 'hill_search']
    METRICS = ['x_diff', 'f_diff']
    PLOT_N_BEST = 2
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(ncols=n_metrics, nrows=1, figsize=(n_metrics * 4, 4))
    for k, method in enumerate(METHODS):
        df_plot = df_agg.loc[df_agg['method'] == method]
        df_plot.sort_values(by=('x_diff', 'mean'), ascending=True, inplace=True)

        if df_plot.empty:
            continue

        for i, metric in enumerate(METRICS):
            for j in range(PLOT_N_BEST):
                row = df_plot.iloc[j]
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
            axes[i].set_xticks(np.arange(len(METHODS)))
            axes[i].set_xticklabels(METHODS)
            axes[i].set_xlim((-0.5, len(METHODS) - 0.5))
            axes[i].set_yscale('log')

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiBS run')
    parser.add_argument('--exp_name', type=str, default='test_may23')
    args = parser.parse_args()
    main(args)