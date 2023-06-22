import pandas as pd
import numpy as np
import argparse

from typing import Tuple
from matplotlib import pyplot as plt
from experiments.util import collect_exp_results, ucb, lcb, median, count


def different_method_plot(df_agg: pd.DataFrame, metric: str = 'l2_dist', display: bool = True) \
        -> Tuple[pd.DataFrame, plt.Figure]:
    models = set(df_agg['score_estim'])

    best_rows = []
    for model in models:
        df_agg_model = df_agg.loc[df_agg['score_estim'] == model]
        df_agg_model.sort_values(by=(metric, 'mean'), ascending=True, inplace=True)
        best_rows.append(df_agg_model.iloc[0])

    best_rows_df = pd.DataFrame(best_rows)

    fig, ax = plt.subplots()
    x_pos = 2 * np.arange(len(models))
    ax.bar(x_pos, best_rows_df[(metric, 'mean')], color='#ADD8E6')
    ax.errorbar(x_pos, best_rows_df[(metric, 'mean')], yerr=best_rows_df[(metric, 'std')],
                linestyle='', capsize=4., color='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(best_rows_df['score_estim'], rotation=-20)
    ax.set_ylabel(metric)
    fig.tight_layout()

    if display:
        plt.show()
        print(best_rows_df.to_string())

    return best_rows_df, fig



def main(args):
    df_full, param_names = collect_exp_results(exp_name=args.exp_name)

    df_full['bandwidth'] = df_full['bandwidth'].fillna(value='auto')
    df_full.loc[df_full['bandwidth'] == 'auto', 'score_estim'] = df_full['score_estim'] + '_auto'

    result_is_nan = df_full['fisher_div'].isna()
    df_full_dropped = df_full[result_is_nan]
    print(f'Dropped results due to NaNs: {df_full_dropped.shape[0]}')
    df_full = df_full[~result_is_nan]


    # group over everything except seeds and aggregate over the seeds
    groupby_names = list(set(param_names) - {'model_seed', 'data_seed'})

    # # first take mean over the data seeds
    # df_mean = df_full.groupby(by=groupby_names + ['model_seed'], axis=0).mean()
    # df_mean.reset_index(drop=False, inplace=True)

    # then compute the stats over the model seeds
    df_agg = df_full.groupby(by=groupby_names, axis=0).aggregate(['mean', 'std', count], axis=0)# ucb, lcb, median, count], axis=0)
    df_agg.reset_index(drop=False, inplace=True)
    # df_agg.sort_values(by=[('nll', 'mean')], ascending=True, inplace=True)

    # filter all the rows where the count is less than 3
    df_agg = df_agg[df_agg['fisher_div']['count'] >= 3]

    #df_agg = df_agg[df_agg['num_samples'] == 256 * 4]
    different_method_plot(df_agg[(df_agg['num_samples'] == 256 * 4)], metric='cos_dist', display=True)

    df_method = df_agg[(df_agg['score_estim'] == 'ssge') & (df_agg['num_samples'] == 256 * 4)
                       & (df_agg['num_dim'] == 16)]
    fig, ax = plt.subplots(ncols=2)
    for i, param in enumerate(['bandwidth', 'eta_ssge']):
        ax[i].scatter(df_method[param], df_method[('l2_dist', 'mean')])
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
    fig.show()

    df_nu = df_agg[(df_agg['score_estim'] == 'nu_method') & (df_agg['num_samples'] == 256 * 2) &
                   (df_agg['num_dim'] == 16)]
    plt.scatter(df_nu['bandwidth'], df_nu[('l2_dist', 'mean')])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


    print('Modles:', set(df_agg['model']))

    different_method_plot(df_agg, metric='nll')
    different_method_plot(df_agg, metric='rmse')

    df_agg = df_agg[df_agg['model'] == 'BNN_FSVGD_SimPrior_ssge'][df_agg['bandwidth_score_estim'] != 'auto']

    metric = 'nll'
    for param in ['bandwidth_svgd', 'num_train_steps', 'num_measurement_points', 'num_f_samples', 'bandwidth_score_estim']:
        plt.scatter(df_agg[param], df_agg[(metric, 'mean')])
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.show()

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
    parser = argparse.ArgumentParser(description='Inspect results of a regression experiment.')
    parser.add_argument('--exp_name', type=str, default='mar31_score_estim')
    args = parser.parse_args()
    main(args)