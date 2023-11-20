from experiments.util import (generate_run_commands, generate_base_command, RESULT_DIR, sample_param_flags, hash_dict)
from experiments.data_provider import DATASET_CONFIGS

import experiments.meta_learning_exp.run_meta_learning_exp
import numpy as np
import datetime
import itertools
import argparse
import os

MODEL_SPECIFIC_CONFIG = {
    'PACOH': {
        'prior_weight': {'distribution': 'log_uniform_10', 'min': -1., 'max': 0.},
        'num_iter_meta_train': {'values': [20000, 30000, 40000]},
        'meta_batch_size': {'values': [4, 8, 16]},
        'bandwidth': {'distribution': 'log_uniform_10', 'min': 0., 'max': 2.},
        'lr': {'distribution': 'log_uniform_10', 'min': -4., 'max': -3}
    },
    'NP': {
        'num_iter_meta_train': {'values': [60000]},
        'latent_dim': {'values': [64, 128, 256]},
        'hidden_dim': {'values': [32, 64, 128]},
        'lr': {'distribution': 'log_uniform_10', 'min': -4., 'max': -3},
    },
}


def main(args):
    # setup random seeds
    rds = np.random.RandomState(args.seed)
    model_seeds = list(rds.randint(0, 10**6, size=(100,)))
    data_seeds = list(rds.randint(0, 10**6, size=(100,)))

    sweep_config = {
        'learn_likelihood_std': {'value': args.learn_likelihood_std},
        'data_source': {'value': args.data_source},
        'num_samples_train': DATASET_CONFIGS[args.data_source]['num_samples_train'],
        'model': {'value': args.model},
    }
    # update with model specific sweep ranges
    assert args.model in MODEL_SPECIFIC_CONFIG
    sweep_config.update(MODEL_SPECIFIC_CONFIG[args.model])

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
            cmd = generate_base_command(experiments.meta_learning_exp.run_meta_learning_exp,
                                        flags=dict(**flags, **{'model_seed': model_seed, 'data_seed': data_seed}))
            command_list.append(cmd)
            output_file_list.append(os.path.join(exp_result_folder, f'{model_seed}_{data_seed}.out'))

    print(command_list[0])
    generate_run_commands(command_list, output_file_list, num_cpus=args.num_cpus,
                          num_gpus=1 if args.gpu else 0, mode=args.run_mode, prompt=not args.yes)


if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # sweep args
    parser.add_argument('--num_hparam_samples', type=int, default=20)
    parser.add_argument('--num_model_seeds', type=int, default=3, help='number of model seeds per hparam')
    parser.add_argument('--num_data_seeds', type=int, default=4, help='number of model seeds per hparam')
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--run_mode', type=str, default='euler')

    # general args
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--seed', type=int, default=94563)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--yes', default=False, action='store_true')

    # data parameters
    parser.add_argument('--data_source', type=str, default='pendulum')

    # # standard BNN parameters
    parser.add_argument('--model', type=str, default='NP')
    parser.add_argument('--learn_likelihood_std', type=int, default=0)

    args = parser.parse_args()
    main(args)
