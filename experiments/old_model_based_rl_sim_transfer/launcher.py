import exp
from experiments.util import generate_run_commands, generate_base_command

PROJECT_NAME = 'ModelBasedRLSimTransferComparison'

applicable_configs = {
    'horizon_len': [8, 16, 32],
    'seed': [0, 1],
    'project_name': [PROJECT_NAME],
    'num_episodes': [40],
    'sac_num_env_steps': [1_000_000, 2_000_000],
    'bnn_train_steps': [20_000, 40_000],
    'learnable_likelihood_std': ['yes', 'no'],
    'reset_bnn': ['no'],
    'use_sim_prior': [0, 1],
    'include_aleatoric_noise': [1],
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
    for seed in applicable_configs['seed']:
        for project_name in applicable_configs['project_name']:
            for horizon_len in applicable_configs['horizon_len']:
                for num_episodes in applicable_configs['num_episodes']:
                    for sac_num_env_steps in applicable_configs['sac_num_env_steps']:
                        for bnn_train_steps in applicable_configs['bnn_train_steps']:
                            for learnable_likelihood_std in applicable_configs['learnable_likelihood_std']:
                                for reset_bnn in applicable_configs['reset_bnn']:
                                    for use_sim_prior in applicable_configs['use_sim_prior']:
                                        for include_aleatoric_noise in applicable_configs['include_aleatoric_noise']:
                                            flags = {
                                                'sac_num_env_steps': sac_num_env_steps,
                                                'bnn_train_steps': bnn_train_steps,
                                                'horizon_len': horizon_len,
                                                'seed': seed,
                                                'project_name': project_name,
                                                'num_episodes': num_episodes,
                                                'learnable_likelihood_std': learnable_likelihood_std,
                                                'reset_bnn': reset_bnn,
                                                'use_sim_prior': use_sim_prior,
                                                'include_aleatoric_noise': include_aleatoric_noise,
                                            }

                                            cmd = generate_base_command(exp, flags=flags)
                                            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
