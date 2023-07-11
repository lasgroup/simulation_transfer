import bimodal_exp
from experiments.util import generate_run_commands, generate_base_command

PROJECT_NAME = 'LinearBimodalSimPrior'

applicable_configs = {
    'score_estimator': ['gp', 'ssge', 'nu-method', 'kde'],
    'seed': [i for i in range(10)],
}


def main():
    command_list = []
    for score_estimator in applicable_configs['score_estimator']:
        for seed in applicable_configs['seed']:
            flags = {
                'seed': seed,
                'score_estimator': score_estimator,
                'project_name': PROJECT_NAME
            }

            cmd = generate_base_command(bimodal_exp, flags=flags)
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', promt=True, mem=16000)


if __name__ == '__main__':
    main()
