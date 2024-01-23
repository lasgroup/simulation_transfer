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
        'prior_weight': {'values': [0.1]},
        'hyper_prior_weight': {'values': [1e-3]},
        'num_iter_meta_train': {'values': [100_000]},
        'meta_batch_size': {'values': [4]},
        'bandwidth': {'values': [10.]},
        'lr': {'values': [5e-4]},
    },
    'NP': {
        'num_iter_meta_train': {'values': [100_000]},
        'latent_dim': {'values': [256]},
        'hidden_dim': {'values': [128]},
        'lr': {'values': [5e-4]},
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
        'pred_diff': {'value': args.pred_diff},
        'model': {'value': args.model},
    }
    # update with model specific sweep ranges
    assert args.model in MODEL_SPECIFIC_CONFIG
    sweep_config.update(MODEL_SPECIFIC_CONFIG[args.model])

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, f'{args.data_source}_{args.model}')

    if args.data_source == 'racecar':
        N_SAMPLES_LIST = [50, 100, 200, 400, 800, 1600, 3200]
    elif args.data_source == 'pendulum':
        N_SAMPLES_LIST = [10, 20, 40, 80, 160, 320, 640]
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
                cmd = generate_base_command(experiments.meta_learning_exp.run_meta_learning_exp,
                                            flags=dict(**flags, **{'model_seed': model_seed, 'data_seed': data_seed,
                                                                   'num_samples_train': num_samples_train}))
                command_list.append(cmd)
                output_file_list.append(os.path.join(exp_result_folder, f'{model_seed}_{data_seed}.out'))

    generate_run_commands(command_list, output_file_list, num_cpus=args.num_cpus, mem=8*1024,
                          duration='11:59:00' if args.long else '3:59:00',
                          num_gpus=1 if args.gpu else 0, mode=args.run_mode, prompt=not args.yes)


if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # sweep args
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_model_seeds', type=int, default=5, help='number of model seeds per hparam')
    parser.add_argument('--num_data_seeds', type=int, default=5, help='number of model seeds per hparam')
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--run_mode', type=str, default='euler')

    # general args
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--seed', type=int, default=94563)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--yes', default=False, action='store_true')
    parser.add_argument('--long', default=False, action='store_true')

    # data parameters
    parser.add_argument('--data_source', type=str, default='racecar')
    parser.add_argument('--pred_diff', type=int, default=1)

    # # standard BNN parameters
    parser.add_argument('--model', type=str, default='BNN_SVGD')
    parser.add_argument('--learn_likelihood_std', type=int, default=0)

    args = parser.parse_args()
    main(args)
