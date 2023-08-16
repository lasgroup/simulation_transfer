import hyperparams_exp
from experiments.util import generate_run_commands, generate_base_command

PROJECT_NAME = 'RaceCarPPOHyperparamsN2'

applicable_configs = {
    'num_envs': [32, ],
    'lr': [3e-4, 1e-3],
    'entropy_cost': [1e-5, 1e-3, 1e-1],
    'unroll_length': [32, 128, 200],
    'batch_size': [64, 128],
    'num_minibatches': [16, 64, 128],
    'num_updates_per_batch': [1, 4, 8],
    'net_arch': ['small', 'medium'],
    'seed': [0, 1, 2],
    'project_name': [PROJECT_NAME],
}


# applicable_configs = {
#     'num_envs': [32, ],
#     'lr': [3e-4],
#     'entropy_cost': [1e-3],
#     'unroll_length': [200],
#     'batch_size': [128],
#     'num_minibatches': [64],
#     'num_updates_per_batch': [4],
#     'net_arch': ['small'],
#     'seed': [0],
#     'project_name': ['TestRunCarPPOHyperparams'],
# }


def main():
    command_list = []
    for num_envs in applicable_configs['num_envs']:
        for lr in applicable_configs['lr']:
            for entropy_cost in applicable_configs['entropy_cost']:
                for unroll_length in applicable_configs['unroll_length']:
                    for batch_size in applicable_configs['batch_size']:
                        for num_minibatches in applicable_configs['num_minibatches']:
                            for num_updates_per_batch in applicable_configs['num_updates_per_batch']:
                                for net_arch in applicable_configs['net_arch']:
                                    for seed in applicable_configs['seed']:
                                        for project_name in applicable_configs['project_name']:
                                            flags = {
                                                'num_envs': num_envs,
                                                'lr': lr,
                                                'entropy_cost': entropy_cost,
                                                'unroll_length': unroll_length,
                                                'batch_size': batch_size,
                                                'num_minibatches': num_minibatches,
                                                'num_updates_per_batch': num_updates_per_batch,
                                                'net_arch': net_arch,
                                                'seed': seed,
                                                'project_name': project_name
                                            }

                                            cmd = generate_base_command(hyperparams_exp, flags=flags)
                                            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', promt=True, mem=16000)


if __name__ == '__main__':
    main()
