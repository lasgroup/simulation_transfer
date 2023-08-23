import exp
from experiments.util import generate_run_commands, generate_base_command

PROJECT_NAME = 'CheckAdvantageOfSimPrior'

applicable_configs = {
    'num_train_traj': [1, 2, 4, 10],
    'use_sim_prior': [0, 1],
}


def main():
    command_list = []
    for num_train_traj in applicable_configs['num_train_traj']:
        for use_sim_prior in applicable_configs['use_sim_prior']:
            flags = {
                'num_train_traj': num_train_traj,
                'use_sim_prior': use_sim_prior,
            }

            cmd = generate_base_command(exp, flags=flags)
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
