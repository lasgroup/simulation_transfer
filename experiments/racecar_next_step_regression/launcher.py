import exp
from experiments.util import generate_run_commands, generate_base_command

PROJECT_NAME = 'CheckAdvantageSimPriorNextStep_N1'

applicable_configs = {
    'num_train_traj': [1, 2, 4, 10],
    'use_sim_prior': [0, 1],
    'project_name': [PROJECT_NAME],
    'learn_std': [0, 1],
}


def main():
    command_list = []
    for num_train_traj in applicable_configs['num_train_traj']:
        for use_sim_prior in applicable_configs['use_sim_prior']:
            for project_name in applicable_configs['project_name']:
                for lean_std in applicable_configs['learn_std']:
                    flags = {
                        'project_name': project_name,
                        'num_train_traj': num_train_traj,
                        'use_sim_prior': use_sim_prior,
                        'learn_std': lean_std,
                    }

                    cmd = generate_base_command(exp, flags=flags)
                    command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
