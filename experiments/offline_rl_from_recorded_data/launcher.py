import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'OfflineTrainingFromRecordedDataFixedGoalAndStart'

applicable_configs = {
    'horizon_len': [2 ** 6, 100, 200],
    'seed': [0, 1],
    'project_name': [PROJECT_NAME],
    'sac_num_env_steps': [1_000_000, 2_000_000, 3_000_000, 10_000_000],
    'bnn_train_steps': [40_000, 60_000, 80_000],
    'learnable_likelihood_std': ['yes'],
    'include_aleatoric_noise': [1],
    'best_bnn_model': [1],
    'best_policy': [1],
    'margin_factor': [20.0],
    'predict_difference': [1],
}


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
