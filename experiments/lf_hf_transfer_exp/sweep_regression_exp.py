from experiments.util import (generate_run_commands, generate_base_command, RESULT_DIR, sample_param_flags, hash_dict)
from experiments.data_provider import DATASET_CONFIGS

import experiments.lf_hf_transfer_exp.run_regression_exp
import numpy as np
import datetime
import itertools
import argparse
import os

MODEL_SPECIFIC_CONFIG = {
    'BNN_SVGD': {
        'bandwidth_svgd': {'distribution': 'log_uniform', 'min': -1., 'max': 4.},
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
    },
    'BNN_FSVGD': {
        'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
        'bandwidth_gp_prior': {'distribution': 'log_uniform', 'min': -2., 'max': 0.},
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [16, 32, 64, 128]},
    },
    'BNN_FSVGD_SimPrior_gp': {
        'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [32]},
        'num_f_samples': {'values': [1024]},
        'added_gp_lengthscale': {'distribution': 'uniform', 'min': 2., 'max': 10.},   # racecar: 4 - 8  # pendulum (diff): > 8 - 15.
        #'added_gp_outputscale': {'distribution': 'uniform', 'min': 0.05, 'max': 0.5},  # racecar: 0.2 - 0.8
        'added_gp_outputscale': {'distribution': 'uniform', 'min': 0.5, 'max': 4.},  # pendulum (diff): 1 - 3.0
    },
    'BNN_FSVGD_SimPrior_nu-method': {
        'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [32]},
        'num_f_samples': {'values': [512]},
        'bandwidth_score_estim': {'distribution': 'uniform', 'min': 0.8, 'max': 2.0},
        'added_gp_lengthscale': {'distribution': 'uniform', 'min': 2., 'max': 10.},
        'added_gp_outputscale': {'distribution': 'uniform', 'min': 0.5, 'max': 4.},
    },
    'BNN_FSVGD_SimPrior_ssge': {
        'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [8, 16, 32]},
        'num_f_samples': {'values': [512]},
        'bandwidth_score_estim': {'distribution': 'log_uniform_10', 'min': -0.5, 'max': 1.},
    },
    'BNN_FSVGD_SimPrior_gp+nu-method': {
        'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [16, 32]},
        'num_f_samples': {'values': [512]},
        'switch_score_estimator_frac': {'values': [0.6667]},
        'bandwidth_score_estim': {'distribution': 'log_uniform_10', 'min': 0.0, 'max': 0.5},
    },
    'BNN_FSVGD_SimPrior_kde': {
        'bandwidth_svgd': {'distribution': 'log_uniform', 'min': -2., 'max': 2.},
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [16, 32]},
        'num_f_samples': {'values': [512, 1024, 2056]},
    },
    'BNN_MMD_SimPrior': {
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [8, 16, 32, 64]},
        'num_f_samples': {'values': [64, 128, 256, 512]},
    },
    'BNN_SVGD_DistillPrior': {
        'bandwidth_svgd': {'distribution': 'log_uniform', 'min': -2., 'max': 2.},
        'min_train_steps': {'values': [2500]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [8, 16, 32]},
        'num_f_samples': {'values': [64, 128, 256]},
        'num_distill_steps': {'values': [30000, 60000]},
    },
}


def main(args):
    # setup random seeds
    rds = np.random.RandomState(args.seed)
    model_seeds = list(rds.randint(0, 10**6, size=(100,)))
    data_seeds = list(rds.randint(0, 10**6, size=(100,)))

    sweep_config = {
        'data_source': {'value': args.data_source},
        'num_samples_train': DATASET_CONFIGS[args.data_source]['num_samples_train'],
        'model': {'value': args.model},
        'learn_likelihood_std': {'value': args.learn_likelihood_std},
        #'likelihood_std': {'value': None},
        'num_particles': {'value': 20},
        'data_batch_size': {'value': 8},
        'pred_diff': {'value': args.pred_diff},
    }
    # update with model specific sweep ranges
    model_name = args.model.replace('_no_add_gp', '')
    assert model_name in MODEL_SPECIFIC_CONFIG
    sweep_config.update(MODEL_SPECIFIC_CONFIG[model_name])

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, f'{args.data_source}_{args.model}')

    command_list = []
    output_file_list = []
    for _ in range(args.num_hparam_samples):
        flags = sample_param_flags(sweep_config)

        exp_result_folder = os.path.join(exp_path, hash_dict(flags))
        flags['exp_result_folder'] = exp_result_folder

        for model_seed, data_seed in itertools.product(model_seeds[:args.num_model_seeds],
                                                       data_seeds[:args.num_data_seeds]):
            cmd = generate_base_command(experiments.lf_hf_transfer_exp.run_regression_exp,
                                        flags=dict(**flags, **{'model_seed': model_seed, 'data_seed': data_seed}))
            command_list.append(cmd)
            output_file_list.append(os.path.join(exp_result_folder, f'{model_seed}_{data_seed}.out'))

    generate_run_commands(command_list, output_file_list, num_cpus=args.num_cpus,
                          num_gpus=1 if args.gpu else 0, mode=args.run_mode, prompt=not args.yes)


if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # sweep args
    parser.add_argument('--num_hparam_samples', type=int, default=20)
    parser.add_argument('--num_model_seeds', type=int, default=3, help='number of model seeds per hparam')
    parser.add_argument('--num_data_seeds', type=int, default=3, help='number of model seeds per hparam')
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--run_mode', type=str, default='euler')

    # general args
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--seed', type=int, default=94563)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--yes', default=False, action='store_true')

    # data parameters
    parser.add_argument('--data_source', type=str, default='pendulum_hf')
    parser.add_argument('--pred_diff', type=int, default=1)

    # # standard BNN parameters
    parser.add_argument('--model', type=str, default='BNN_SVGD')
    parser.add_argument('--learn_likelihood_std', type=int, default=0)

    args = parser.parse_args()
    main(args)
