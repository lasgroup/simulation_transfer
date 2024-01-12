import pandas as pd
import numpy as np
import argparse

from typing import Tuple
from matplotlib import pyplot as plt
from experiments.util import collect_exp_results, ucb, lcb, median, count


def plot_cos_dist_vs_num_samples(df: pd.DataFrame, plot_title: str = ""):
    score_estimators = df['score_estim'].unique()

    # Create the figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    fig.suptitle(plot_title)

    for score_estim in score_estimators:
        # Filter the DataFrame based on the score estimator
        df_filtered = df[df['score_estim'] == score_estim]

        # Flatten the column names to a single level
        df_filtered.columns = [(col[0] if col[1] == '' else '_'.join(col)) for col in df_filtered.columns]

        # Find the indices of rows with the lowest 'cos_dist_mean' for each unique number of samples
        best_indices = df_filtered.groupby('num_samples')['cos_dist_mean'].idxmin()

        # Filter the DataFrame to include only the best hyperparameter configurations
        df_best_configs = df_filtered.loc[best_indices]

        # Sort the filtered DataFrame based on the 'num_samples' column
        df_best_configs = df_best_configs.sort_values(by='num_samples')

        # Extract the relevant data
        num_samples = df_best_configs['num_samples']
        cos_dist_mean = df_best_configs['cos_dist_mean']
        cos_dist_std = df_best_configs['cos_dist_std']
        l2_dist_mean = df_best_configs['l2_dist_mean']
        l2_dist_std = df_best_configs['l2_dist_std']

        # Plot the data with error bars for Cosine Distance
        axes[0].errorbar(num_samples, cos_dist_mean, yerr=cos_dist_std, marker='o', capsize=4, label=score_estim)
        axes[0].set_xlabel('Number of Samples')
        axes[0].set_ylabel('Cosine Distance')
        axes[0].set_title('Cosine Distance vs. Number of Samples')

        # Plot the data with error bars for L2 Distance
        axes[1].errorbar(num_samples, l2_dist_mean, yerr=l2_dist_std, marker='o', capsize=4, label=score_estim)
        axes[1].set_xlabel('Number of Samples')
        axes[1].set_ylabel('L2 Distance')
        axes[1].set_title('L2 Distance vs. Number of Samples')

    # Add the legend
    axes[0].legend()
    axes[1].legend()

    # Show the plot
    plt.show()

    return fig, axes


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

    # then compute the stats over the model seeds
    df_agg = df_full.groupby(by=groupby_names, axis=0).aggregate(['mean', 'std', count], axis=0)# ucb, lcb, median, count], axis=0)
    df_agg.reset_index(drop=False, inplace=True)

    # filter all the rows where the count is less than 3
    df_agg = df_agg[df_agg['fisher_div']['count'] >= 3]

    # filter all the rows where the count is less than 3

    for num_dim in df_agg['num_dim'].unique():
        fig, axes = plot_cos_dist_vs_num_samples(
            df_agg[(df_agg['num_dim'] == num_dim) & (df_agg['dist_type'] == 'gp')],
            plot_title=f'Score estimation for GP marginals, num_dim={num_dim}')
        fig_name = f'score_estim_num_samples_gp_{num_dim}'
        fig_path = f'plots/{fig_name}'
        fig.savefig(f'{fig_path}.pdf')
        fig.savefig(f'{fig_path}.png')
        print('Saved figure to', fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect results of a regression experiment.')
    parser.add_argument('--exp_name', type=str, default='mar31_score_estim')
    args = parser.parse_args()
    main(args)