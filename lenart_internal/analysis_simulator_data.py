import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def to_pretty_group_name(group_name: str):
    if 'use_grey_box=1' in group_name:
        return 'Grey box'
    elif 'use_sim_prior=0' in group_name:
        return 'No sim prior'
    elif 'use_sim_prior=1' in group_name and 'high_fidelity=0' in group_name:
        return 'Low fidelity'
    elif 'use_sim_prior=1' in group_name and 'high_fidelity=1' in group_name:
        return 'High fidelity'


def join_group_names_and_prepare_statistics_mean(data: pd.DataFrame):
    # Summary of rewards
    summary_rewards = data.groupby('Group')['reward_mean_on_simulator'].agg(['mean', 'std'])

    summary = pd.concat([summary_rewards], axis=1)
    summary.reset_index(inplace=True)
    summary.columns = ['group_name', 'mean', 'std', ]
    return summary


def join_group_names_and_prepare_statistics(data: pd.DataFrame):
    # Summary of rewards
    # summary_rewards = data.groupby('Group')['reward_mean_on_simulator'].agg(['mean', 'std'])
    #
    # summary = pd.concat([summary_rewards], axis=1)
    # summary.reset_index(inplace=True)
    # summary.columns = ['group_name', 'mean', 'std', ]
    # Summary of rewards
    summary_rewards = data.groupby('Group')['reward_mean_on_simulator'].agg(['median',
                                                                             lambda x: x.quantile(0.2),
                                                                             # 0.2 quantile
                                                                             lambda x: x.quantile(0.8)
                                                                             # 0.8 quantile
                                                                             ])

    summary = pd.concat([summary_rewards], axis=1)
    summary.reset_index(inplace=True)
    summary.columns = ['group_name',
                       'median_rewards', '0.2_quantile_rewards', '0.8_quantile_rewards']
    return summary


def plot_rewards_mean(data: pd.DataFrame, max_offline_data: int | None = None):
    offline_transitions = data['num_offline_collected_transitions'].unique()
    offline_transitions.sort()

    means = []
    stds = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for index, offline_transition in enumerate(offline_transitions):
        cur_data = data.loc[data['num_offline_collected_transitions'] == offline_transition]
        summary = join_group_names_and_prepare_statistics_mean(cur_data)

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
        ax.plot(offline_transitions, means[:, index], label=group)
        ax.fill_between(offline_transitions,
                        means[:, index] - stds[:, index],
                        means[:, index] + stds[:, index], alpha=0.2)
    ax.set_title(f'Mean +- std')
    ax.set_xlabel('Number of offline transitions')
    ax.set_ylabel('Reward')
    ax.legend()
    plt.savefig('offline_rl_simulated_data.pdf')
    plt.show()


def plot_rewards(data: pd.DataFrame, max_offline_data: int | None = None):
    offline_transitions = data['num_offline_collected_transitions'].unique()
    offline_transitions.sort()

    medians = []
    quantile_20 = []
    quantile_80 = []
    group_names = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for index, offline_transition in enumerate(offline_transitions):
        cur_data = data.loc[data['num_offline_collected_transitions'] == offline_transition]
        summary = join_group_names_and_prepare_statistics(cur_data)

        medians.append(summary['median_rewards'].to_numpy())
        quantile_20.append(summary['0.2_quantile_rewards'].to_numpy())
        quantile_80.append(summary['0.8_quantile_rewards'].to_numpy())
        if index == 0:
            group_names = list(map(lambda x: to_pretty_group_name(x), summary['group_name']))

    medians = np.stack(medians, axis=0)
    quantile_20 = np.stack(quantile_20, axis=0)
    quantile_80 = np.stack(quantile_80, axis=0)
    if max_offline_data:
        idx = offline_transitions <= max_offline_data
        offline_transitions = offline_transitions[idx]
        medians = medians[idx]
        quantile_20 = quantile_20[idx]
        quantile_80 = quantile_80[idx]

    for index, group in enumerate(group_names):
        ax.plot(offline_transitions, medians[:, index], label=group)
        ax.fill_between(offline_transitions, quantile_20[:, index], quantile_80[:, index], alpha=0.2)
    ax.set_title(f'Median and 0.2-0.8 confidence interval')
    ax.set_xlabel('Number of offline transitions')
    ax.set_ylabel('Reward')
    ax.legend()
    plt.savefig('offline_rl_simulated_data.pdf')
    plt.show()


if __name__ == '__main__':
    max_offline_data = 4_000
    data = pd.read_csv('wandb_runs.csv')
    plot_rewards_mean(data[data.bandwidth_svgd == 0.05], max_offline_data=max_offline_data)
