import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'FirstExperimentDelay'

applicable_configs = {
    'horizon_len': [2 ** 6],
    'seed': [0, 1],
    'project_name': [PROJECT_NAME],
    'num_episodes': [40],
    'sac_num_env_steps': [1_000_000],
    'bnn_train_steps': [20_000, 40_000],
    'learnable_likelihood_std': ['yes', 'no'],
    'reset_bnn': ['no', 'yes'],
    'use_sim_prior': [0],
    'include_aleatoric_noise': [1],
    'best_bnn_model': [1],
    'best_policy': [1],
    'margin_factor': [20.0],
    'predict_difference': [1],
    'num_frame_stack': [2, 3],
    'delay': [0.09],
}


# applicable_configs = {
#     'horizon_len': [8],
#     'seed': [0],
#     'project_name': ['TestModelBasedRLSimTransfer'],
#     'num_episodes': [40],
#     'sac_num_env_steps': [2000],
#     'bnn_train_steps': [2000],
#     'learnable_likelihood_std': ['no'],
#     'reset_bnn': ['no'],
#     'use_sim_prior': [0],
#     'include_aleatoric_noise': [1],
# }


def main():
    command_list = []

    all_flags_combinations = dict_permutations(applicable_configs)
    for flags in all_flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
