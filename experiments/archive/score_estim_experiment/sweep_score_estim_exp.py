from experiments.util import (generate_run_commands, generate_base_command, RESULT_DIR, sample_param_flags, hash_dict)

import experiments.score_estim_experiment.run_score_matching_exp
import numpy as np
import datetime
import itertools
import argparse
import os


MODEL_SPECIFIC_CONFIG = {
    'ssge': {
        'bandwidth': {'distribution': 'log_uniform', 'min': -1., 'max': 1.},
        'add_linear_kernel': {'values': [0, 1]},
    },
    'ssge_auto': {
        'add_linear_kernel': {'values': [0, 1]},
    },
    'kde': {
        'bandwidth': {'distribution': 'log_uniform', 'min': -2., 'max': 2.},
    },
    'kde_auto': {},
    'nu_method': {
        'bandwidth': {'distribution': 'log_uniform_10', 'min': -1, 'max': 1.5},
        'lambda_nu': {'distribution': 'log_uniform_10', 'min': -4, 'max': -2},
    },
}

DATASET_CONFIGS = {
    'sinusoids1d': {
        'likelihood_std': {'value': 0.1},
        'num_samples_train': {'value': 5},
    },
    'sinusoids2d': {
        'likelihood_std': {'value': 0.1},
        'num_samples_train': {'value': 5},
    },
    'pendulum': {
        'likelihood_std': {'value': 0.02},
        'num_samples_train': {'value': 20},
    }
}

NUM_SAMPLES = [64, 128, 256, 512, 1024, 2048]
NUM_DIMS = [2, 4, 8, 16, 32, 64]


def main(args):
    # setup random seeds
    assert args.score_estim in MODEL_SPECIFIC_CONFIG
    sweep_config = MODEL_SPECIFIC_CONFIG[args.score_estim]

    score_estim = args.score_estim[:-5] if args.score_estim.endswith('_auto') else args.score_estim

    sweep_config .update({
        'dist_type': {'value': args.dist_type},
        'score_estim': {'value': score_estim},
        'num_data_seeds': {'value': 10},
    })
    # update with model specific sweep ranges

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, f'score_estim_{args.dist_type}_{args.score_estim}')

    command_list = []
    output_file_list = []
    for _ in range(args.num_hparam_samples):
        flags = sample_param_flags(sweep_config)

        exp_result_folder = os.path.join(exp_path, hash_dict(flags))
        flags['exp_result_folder'] = exp_result_folder

        for num_dim, num_samples in itertools.product(NUM_DIMS, NUM_SAMPLES):
            cmd = generate_base_command(experiments.score_estim_experiment.run_score_matching_exp,
                                        flags=dict(**flags, **{'num_dim': num_dim, 'num_samples': num_samples}))
            command_list.append(cmd)
            output_file_list.append(os.path.join(exp_result_folder, f'{num_dim}_{num_samples}.out'))

    generate_run_commands(command_list, output_file_list, num_cpus=args.num_cpus, mode=args.run_mode, prompt=True,
                          duration='0:20:00')


if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # sweep args
    parser.add_argument('--num_hparam_samples', type=int, default=20)
    parser.add_argument('--num_model_seeds', type=int, default=1, help='number of model seeds per hparam')
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--run_mode', type=str, default='euler')

    # general args
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--seed', type=int, default=94563)

    parser.add_argument('--dist_type', type=str, default='gp')
    parser.add_argument('--score_estim', type=str, default='ssge_auto')
    # parser.add_argument('--num_dim', type=int, default=2)


    args = parser.parse_args()
    main(args)
