import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'RaceCarSACHyperparamsNX'

applicable_configs = {
    'num_envs': [32, 64, 128, 256],
    'net_arch': ['medium'],
    'seed': [0, 1],
    'project_name': [PROJECT_NAME],
    'batch_size': [64, 128, 256],
    'num_env_steps_between_updates': [16, 32, 64],
    'num_time_steps': [10 ** 6, 3 * 10 ** 6, 10 ** 7],
}


# applicable_configs = {
#     'num_envs': [32, ],
#     'net_arch': ['small'],
#     'seed': [0],
#     'project_name': ['TestRaceCarSACHyperparamsN2'],
#     'batch_size': [32],
#     'max_replay_size': [3 * 10 ** 5, ],
#     'num_env_steps_between_updates': [16],
#     'target_entropy': [-1]
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
