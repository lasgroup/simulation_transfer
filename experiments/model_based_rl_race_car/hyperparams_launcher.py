import hyperparams_exp
from experiments.util import generate_run_commands, generate_base_command

PROJECT_NAME = 'ModelBasedRLN3'

applicable_configs = {
    'horizon_len': [8, 16, 32, 64, 128],
    'seed': [0, 1],
    'project_name': [PROJECT_NAME],
    'num_episodes': [40],
    'sac_num_env_steps': [300_000, 1_000_000, 2_000_000],
    'bnn_train_steps':  [10_000, 20_000],
    'learnable_likelihood_std': ['yes', 'no'],
    'reset_bnn': ['yes', 'no'],
}


# applicable_configs = {
#     'horizon_len': [16],
#     'seed': [0],
#     'project_name': ['TestModelBasedRLN1'],
#     'num_episodes': [5],
#     'sac_num_env_steps': [1000],
#     'bnn_train_steps': [1000],
# }


def main():
    command_list = []
    for seed in applicable_configs['seed']:
        for project_name in applicable_configs['project_name']:
            for horizon_len in applicable_configs['horizon_len']:
                for num_episodes in applicable_configs['num_episodes']:
                    for sac_num_env_steps in applicable_configs['sac_num_env_steps']:
                        for bnn_train_steps in applicable_configs['bnn_train_steps']:
                            for learnable_likelihood_std in applicable_configs['learnable_likelihood_std']:
                                for reset_bnn in applicable_configs['reset_bnn']:
                                    flags = {
                                        'sac_num_env_steps': sac_num_env_steps,
                                        'bnn_train_steps': bnn_train_steps,
                                        'horizon_len': horizon_len,
                                        'seed': seed,
                                        'project_name': project_name,
                                        'num_episodes': num_episodes,
                                        'learnable_likelihood_std': learnable_likelihood_std,
                                        'reset_bnn': reset_bnn
                                    }

                                    cmd = generate_base_command(hyperparams_exp, flags=flags)
                                    command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
