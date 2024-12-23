import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_hyperdata import plotting_constants

LINEWIDTH = 3
SIMULATION = True

if SIMULATION:
    reward_column = 'reward_mean_on_simulator'
    folder = 'simulation'
else:
    reward_column = 'reward_mean_on_learned_model'
    folder = 'real_data'



def to_pretty_group_name(group_name: str):
    if 'use_grey_box=1' in group_name:
        return 'Grey box'
    elif 'use_sim_prior=0' in group_name:
        return 'No sim prior model'
    elif 'use_sim_prior=1' in group_name and 'high_fidelity=0' in group_name:
        return 'Low fidelity model'
    elif 'use_sim_prior=1' in group_name and 'high_fidelity=1' in group_name:
        return 'High fidelity model'


def join_group_names_and_prepare_statistics_mean_SIM(data: pd.DataFrame, ):
    # Summary of rewards
    summary_rewards = data.groupby('Group')[reward_column].agg([
        lambda x: smooth_curve(x, type='mean'),
        lambda x: smooth_curve(x, type='std_err'),
        'max',

    ])

    summary = pd.concat([summary_rewards], axis=1)
    summary.reset_index(inplace=True)
    summary.columns = ['group_name', 'mean', 'std', 'max']
    return summary


def plot_rewards_mean_SIM(data: pd.DataFrame,
                          max_offline_data: int | None = None,
                          sup_title: str = None,
                          width: float = 2.0):
    offline_transitions = data['num_offline_collected_transitions'].unique()
    offline_transitions.sort()

    means = []
    stds = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for index, offline_transition in enumerate(offline_transitions):
        cur_data = data.loc[data['num_offline_collected_transitions'] == offline_transition]
        summary = join_group_names_and_prepare_statistics_mean_SIM(cur_data)

        means.append(summary['mean'].to_numpy())
        stds.append(summary['std'].to_numpy())
        if index == 0:
            group_names = list(map(lambda x: to_pretty_group_name(x), summary['group_name']))

    means = np.stack(means, axis=0)
    stds = np.stack(stds, axis=0)
    if max_offline_data:
        idx = offline_transitions <= max_offline_data
        offline_transitions = offline_transitions[idx]
        means = means[idx]
        stds = stds[idx]

    for index, group in enumerate(group_names):
        if group in plotting_constants.offline_rl_names_transfer.keys():
            method_name = plotting_constants.offline_rl_names_transfer[group]
            ax.plot(offline_transitions, means[:, index],
                    label=method_name,
                    color=plotting_constants.COLORS[method_name],
                    linestyle=plotting_constants.LINE_STYLES[method_name],
                    linewidth=LINEWIDTH)
            ax.fill_between(offline_transitions,
                            means[:, index] - width * stds[:, index],
                            means[:, index] + width * stds[:, index],
                            alpha=0.2,
                            color=plotting_constants.COLORS[method_name], )
    ax.set_title(r'Mean $\pm$ std error')
    ax.set_xlabel('Number of offline transitions')
    ax.set_ylim(0)
    ax.set_ylabel('Reward')
    fig.suptitle(sup_title)
    dir = folder
    if not os.path.exists(dir):
        os.makedirs(dir)
    title = sup_title + 'offline_rl_simulated_data.pdf'
    plt.legend()
    plt.savefig(os.path.join(dir, title))
    plt.show()



if __name__ == '__main__':
    max_offline_data = 5_000
    data = pd.read_csv('wandb_runs_sim_final.csv')
    bandwidth_svgd = data.bandwidth_svgd.unique()
    length_scale_aditive_sim_gp = data.length_scale_aditive_sim_gp.unique()
    for i in bandwidth_svgd:
        for j in length_scale_aditive_sim_gp:
            filtered_data = data[(data.bandwidth_svgd == i) & (data.length_scale_aditive_sim_gp == j)]
            plot_rewards_mean_SIM(filtered_data, max_offline_data=max_offline_data,
                                  sup_title=f'bandwidth_svgd={i}, length_scale_aditive_sim_gp={j}')
