import hyperparams_exp
from experiments.util import generate_run_commands, generate_base_command

PROJECT_NAME = 'RaceCarSACHyperparamsCTRLCost0.005'

applicable_configs = {
    'num_envs': [32, ],
    'net_arch': ['small', 'medium'],
    'seed': [0, 1, 2],
    'project_name': [PROJECT_NAME],
    'batch_size': [32, 64, 128, 256, ],
    'max_replay_size': [5 * 10 ** 4, 10 ** 5, 3 * 10 ** 5, ],
    'num_env_steps_between_updates': [1, 4, 8, 16, 32, 64, 128],
}

# applicable_configs = {
#     'num_envs': [32, ],
#     'net_arch': ['small'],
#     'seed': [0],
#     'project_name': ['TestRaceCarSACHyperparamsN2'],
#     'batch_size': [32],
#     'max_replay_size': [3 * 10 ** 5, ],
#     'num_env_steps_between_updates': [16],
# }


def main():
    command_list = []
    for num_envs in applicable_configs['num_envs']:
        for net_arch in applicable_configs['net_arch']:
            for seed in applicable_configs['seed']:
                for project_name in applicable_configs['project_name']:
                    for batch_size in applicable_configs['batch_size']:
                        for max_replay_size in applicable_configs['max_replay_size']:
                            for num_env_steps_between_updates in applicable_configs['num_env_steps_between_updates']:
                                flags = {
                                    'num_envs': num_envs,
                                    'net_arch': net_arch,
                                    'seed': seed,
                                    'project_name': project_name,
                                    'batch_size': batch_size,
                                    'max_replay_size': max_replay_size,
                                    'num_env_steps_between_updates': num_env_steps_between_updates,
                                }

                                cmd = generate_base_command(hyperparams_exp, flags=flags)
                                command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', prompt=True, mem=16000)


if __name__ == '__main__':
    main()
