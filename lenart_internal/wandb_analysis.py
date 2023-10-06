import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import re

def to_pretty_group_name(group_name: str):
    if 'use_grey_box=1' in group_name:
        return 'Grey box'
    elif 'use_sim_prior=0' in group_name:
        return 'No sim prior'
    elif 'use_sim_prior=1' in group_name and 'high_fidelity=0' in group_name:
        return 'Low fidelity'
    elif 'use_sim_prior=1' in group_name and 'high_fidelity=1' in group_name:
        return 'High fidelity'


def select_runs_for_one_experiment(data: pd.DataFrame, offline_transitions: int, share_of_x0s_in_sac_buffer: float):
    """
    Selects runs for one experiment, that have num_offline_collected_transitions equal to offline_transitions and
    share_of_x0s_in_sac_buffer equal to share_of_x0s_in_sac_buffer.
    """
    return data[(data['num_offline_collected_transitions'] == offline_transitions) &
                (data['share_of_x0s_in_sac_buffer'] == share_of_x0s_in_sac_buffer)]


def join_group_names_and_prepare_statistics(data: pd.DataFrame):
    # Summary of rewards
    summary_rewards = data.groupby('Group')['reward_median_on_pretrained_model'].agg(['median',
                                                                                      lambda x: x.quantile(0.2),
                                                                                      # 0.2 quantile
                                                                                      lambda x: x.quantile(0.8)
                                                                                      # 0.8 quantile
                                                                                      ])

    # Summary of nll
    summary_nll = data.groupby('Group')['eval_on_all_offline_data/nll'].agg(['median',
                                                                             lambda x: x.quantile(0.2),
                                                                             # 0.2 quantile
                                                                             lambda x: x.quantile(0.8)
                                                                             # 0.8 quantile
                                                                             ])

    summary = pd.concat([summary_rewards, summary_nll], axis=1)
    summary.reset_index(inplace=True)
    summary.columns = ['group_name',
                       'median_rewards', '0.2_quantile_rewards', '0.8_quantile_rewards',
                       'median_nll', '0.2_quantile_nll', '0.8_quantile_nll']
    return summary


def plot_statistics_of_offline_rl(data: pd.DataFrame,
                                  figure: plt.Figure | None = None,
                                  share_of_x0s_in_sac_buffer=0.5,
                                  max_offline_data: int | None = None):
    # Get all distinct number of offline transitions

    # offline_transitions = data['num_offline_collected_transitions'].unique()
    offline_transitions = data[data['share_of_x0s_in_sac_buffer'] == share_of_x0s_in_sac_buffer][
        'num_offline_collected_transitions'].unique()
    offline_transitions.sort()
    medians = []
    quantile_20 = []
    quantile_80 = []
    group_names = []

    for index, offline_transition in enumerate(offline_transitions):
        one_exp_data = select_runs_for_one_experiment(data, offline_transitions=offline_transition,
                                                      share_of_x0s_in_sac_buffer=share_of_x0s_in_sac_buffer)
        summary = join_group_names_and_prepare_statistics(one_exp_data)
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

    ax = figure.subplots(nrows=1, ncols=1)

    for index, group in enumerate(group_names):
        ax.plot(offline_transitions, medians[:, index], label=group)
        ax.fill_between(offline_transitions, quantile_20[:, index], quantile_80[:, index], alpha=0.2)
    ax.set_title(f'Median and 0.2-0.8 confidence interval')
    ax.set_xlabel('Number of offline transitions')
    ax.set_ylabel('Reward')
    ax.legend()
    return figure


def plot_statistics_of_nll(data: pd.DataFrame,
                           figure: plt.Figure | None = None,
                           share_of_x0s_in_sac_buffer=0.5,
                           max_offline_data: int | None = None):
    # Get all distinct number of offline transitions
    offline_transitions = data[data['share_of_x0s_in_sac_buffer'] == share_of_x0s_in_sac_buffer][
        'num_offline_collected_transitions'].unique()
    offline_transitions.sort()
    medians = []
    quantile_20 = []
    quantile_80 = []
    group_names = []

    for index, offline_transition in enumerate(offline_transitions):
        one_exp_data = select_runs_for_one_experiment(data, offline_transitions=offline_transition,
                                                      share_of_x0s_in_sac_buffer=share_of_x0s_in_sac_buffer)
        summary = join_group_names_and_prepare_statistics(one_exp_data)
        medians.append(summary['median_nll'].to_numpy())
        quantile_20.append(summary['0.2_quantile_nll'].to_numpy())
        quantile_80.append(summary['0.8_quantile_nll'].to_numpy())
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

    min_value = np.min(quantile_20)
    eps = 1e-1

    medians = medians - min_value + eps
    quantile_20 = quantile_20 - min_value + eps
    quantile_80 = quantile_80 - min_value + eps

    if figure is None:
        figure = plt.figure()

    ax = figure.subplots(nrows=1, ncols=1)

    for index, group in enumerate(group_names):
        ax.plot(offline_transitions, medians[:, index], label=group)
        ax.fill_between(offline_transitions, quantile_20[:, index], quantile_80[:, index], alpha=0.2)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(f'Median and 0.2-0.8 confidence interval')
    ax.set_xlabel('Number of offline transitions')
    ax.set_ylabel(f'Negative Log-Likelihood on 20_000 datapoints \n {min_value:.2f} subtracted')
    ax.legend()
    return figure


def plot_rewards_and_nll(data: pd.DataFrame, share_of_x0s_in_sac_buffer=0.5, max_offline_data: int | None = None,
                         title: str = ''):
    fig = plt.figure(layout='constrained', figsize=(10, 4))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    plot_statistics_of_offline_rl(data, figure=subfigs[0], share_of_x0s_in_sac_buffer=share_of_x0s_in_sac_buffer,
                                  max_offline_data=max_offline_data)
    plot_statistics_of_nll(data, figure=subfigs[1], share_of_x0s_in_sac_buffer=share_of_x0s_in_sac_buffer,
                           max_offline_data=max_offline_data)
    fig.suptitle(title)
    plt.savefig(f'offline_rl_{share_of_x0s_in_sac_buffer}_{title}.pdf')
    plt.show()


def download_runs(project_name: str, likelihood_exponent: float):
    extraction_keys = ['reward_median_on_pretrained_model', 'eval_on_all_offline_data/nll']
    api = wandb.Api()
    runs = api.runs(project_name)
    frames = []
    for run in runs:
        if '_' + str(likelihood_exponent) in run.group:
            nums = re.findall(r"[-+]?(?:\d*\.*\d+)", run.group)
            data = {
                'Group': run.group,
                'share_of_x0s_in_sac_buffer': float(nums[-2]),
                'num_offline_collected_transitions': float(nums[3])
            }
            summary = run.summary
            print(data)
            for key in extraction_keys:
                data[key] = summary[key]
            df = pd.DataFrame(data, index=[0])
            frames.append(df)
    frames = pd.concat(frames, axis=0)
    frames.to_csv('wandb_runs_' + str(likelihood_exponent) + '.csv')


if __name__ == '__main__':
    exponents = [0.25, 0.5, 0.75, 1.0]
    max_offline_data = 2000
    download_data = False
    for exp in exponents:
        if download_data:
            download_runs('sukhijab/OfflineRLLikelihood0_test', likelihood_exponent=exp)
        data = pd.read_csv('wandb_runs_' + str(exp) + '.csv')
        plot_rewards_and_nll(data, share_of_x0s_in_sac_buffer=0.5, max_offline_data=max_offline_data,
                             title='Likelihood exponent ' + str(exp) + ', Max offline data' + str(max_offline_data))
