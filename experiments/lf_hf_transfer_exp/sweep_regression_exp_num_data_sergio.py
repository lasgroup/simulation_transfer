from experiments.util import (generate_run_commands, generate_base_command, RESULT_DIR, sample_param_flags, hash_dict)
from experiments.data_provider import DATASET_CONFIGS

import experiments.lf_hf_transfer_exp.run_regression_exp
from experiments.lf_hf_transfer_exp.run_regression_exp import OUTPUTSCALES_RCCAR
import numpy as np
import datetime
import itertools
import argparse
import os
import jax.numpy as jnp

MODEL_SPECIFIC_CONFIG = {
    'BNN_SVGD': {
        'bandwidth_svgd': {'values': [10.]},
        'min_train_steps': {'values': [5_000]},
        'num_epochs': {'values': [60]},
        'lr': {'values': [1e-3]},
        # 'likelihood_reg': {'values': [10.0]},
    },
    'BNN_FSVGD': {
        'bandwidth_svgd': {'values': [2.0]},
        'min_train_steps': {'values': [5_000]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [32]},
        'num_f_samples': {'values': [1028]},
        'added_gp_lengthscale': {'values': [10.]},
        'added_gp_outputscale': {'values': [0.5]},
        'lr': {'values': [1e-3]},
        'likelihood_reg': {'values': [10.0]},
    },

    'BNN_FSVGD_SimPrior_gp': {
        'bandwidth_svgd': {'values': [2.0]},
        'min_train_steps': {'values': [5_000]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [32]},
        'num_f_samples': {'values': [1028]},
        'added_gp_lengthscale': {'values': [10.]},
        'added_gp_outputscale': {'values': [0.5]},
        'lr': {'values': [1e-3]},
        'likelihood_reg': {'values': [10.0]},
    },

    'BNN_FSVGD_SimPrior_no_add_gp': {
        'bandwidth_svgd': {'values': [10.0]},
        'min_train_steps': {'values': [5_000]},
        'num_epochs': {'values': [200]},
        'num_measurement_points': {'values': [64]},
        'num_f_samples': {'values': [1028]},
        'added_gp_lengthscale': {'values': [1.0]},
        'added_gp_outputscale': {'values': [0.5]},
        'lr': {'values': [1e-3]},
        'likelihood_reg': {'values': [10.0]},
    },

    'SysID': {
    },
    'GreyBox': {
        'bandwidth_svgd': {'values': [2.0]},
        'bandwidth_gp_prior': {'values': [0.4]},
        'min_train_steps': {'values': [5_000]},
        'num_epochs': {'values': [60]},
        'num_measurement_points': {'values': [32]},
        'lr': {'values': [1e-3]},
        'likelihood_reg': {'values': [10.0]},
    },
}


def main(args):
    # setup random seeds
    rds = np.random.RandomState(args.seed)
    model_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))
    data_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))

    sweep_config = {
        'data_source': {'value': args.data_source},
        # 'num_samples_train': DATASET_CONFIGS[args.data_source]['num_samples_train'],
        'model': {'value': args.model},
        'learn_likelihood_std': {'value': args.learn_likelihood_std},
        # 'likelihood_std': {'value': None},
        'num_particles': {'value': 20},
        'data_batch_size': {'value': 8},
        'pred_diff': {'value': args.pred_diff},
        'max_train_steps': {'value': 300_000},
        'num_sim_model_train_steps': {'value': 5_000},
    }
    # update with model specific sweep ranges
    model_name = args.model.replace('_no_add_gp', '')
    model_name = model_name.replace('_hf', '')
    assert model_name in MODEL_SPECIFIC_CONFIG
    sweep_config.update(MODEL_SPECIFIC_CONFIG[model_name])

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, f'{args.data_source}_{args.model}')

    if args.data_source == 'racecar_hf':
        N_SAMPLES_LIST = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    elif args.data_source == 'pendulum_hf':
        N_SAMPLES_LIST = [10, 20, 40, 80, 160, 320, 640, 1280]
    elif args.data_source == 'real_racecar_v3':
        N_SAMPLES_LIST = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    elif args.data_source == 'Sergio_hf':
        N_SAMPLES_LIST = [50, 100, 200, 400, 800, 1600, 3200, 4800, 6400]
    else:
        raise NotImplementedError(f'Unknown data source {args.data_source}.')

    command_list = []
    output_file_list = []
    for _ in range(args.num_hparam_samples):
        flags = sample_param_flags(sweep_config)
        exp_hash = hash_dict(flags)
        for num_samples_train in N_SAMPLES_LIST:
            exp_result_folder = os.path.join(exp_path, f'{exp_hash}_{num_samples_train}')
            flags['exp_result_folder'] = exp_result_folder

            for model_seed, data_seed in itertools.product(model_seeds[:args.num_model_seeds],
                                                           data_seeds[:args.num_data_seeds]):
                cmd = generate_base_command(experiments.lf_hf_transfer_exp.run_regression_exp,
                                            flags=dict(**flags, **{'model_seed': model_seed, 'data_seed': data_seed,
                                                                   'num_samples_train': num_samples_train,
                                                                   }))
                command_list.append(cmd)
                output_file_list.append(os.path.join(exp_result_folder, f'{model_seed}_{data_seed}.out'))

    generate_run_commands(command_list, output_file_list, num_cpus=args.num_cpus,
                          num_gpus=1 if args.gpu else 0, mode=args.run_mode, prompt=not args.yes, duration='23:59:00')


if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # sweep argsx
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_model_seeds', type=int, default=5, help='number of model seeds per hparam')
    parser.add_argument('--num_data_seeds', type=int, default=5, help='number of model seeds per hparam')
    parser.add_argument('--num_cpus', type=int, default=4, help='number of cpus to use')
    parser.add_argument('--run_mode', type=str, default='euler')

    # general args
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--seed', type=int, default=94563)
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--yes', default=False, action='store_true')

    # data parameters
    parser.add_argument('--data_source', type=str, default='pendulum_hf')
    parser.add_argument('--pred_diff', type=int, default=0)

    # # standard BNN parameters
    parser.add_argument('--model', type=str, default='BNN_SVGD')
    parser.add_argument('--learn_likelihood_std', type=int, default=0)

    args = parser.parse_args()
    main(args)
