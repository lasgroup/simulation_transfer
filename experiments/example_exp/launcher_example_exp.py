from experiments.util import generate_base_command, generate_run_commands, hash_dict, sample_flag, RESULT_DIR

import experiments.example_exp.run_example_exp
import argparse
import numpy as np
import copy
import os
import itertools

applicable_configs = {
    'random_search': ['loc', 'scale', 'uniform'],
    'hill_search': ['pertubation_std', 'anneal_factor'],
}

default_configs = {
    'loc': 0.0,
    'scale': 5.0,
    'uniform': False,
    'pertubation_std': 1.0,
    'anneal_factor': 1.0,
}

search_ranges = {
    # random search
    'loc': ['uniform', [-5, 5]],
    'scale': ['loguniform', [np.log10(0.1), np.log10(10.)]],
    'uniform': ['choice', [True, False]],
    # hill climb
    'pertubation_std': ['loguniform', [np.log10(0.1), np.log10(1.)]],
    'anneal_factor': ['loguniform', [np.log10(0.9), np.log10(1.)]],
}

# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}


def main(args):
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10**6, size=(100,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, f'{args.target_fun}_{args.method}_{args.num_samples}')


    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_hparam', 'exp_name', 'num_cpus']]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]

        # determine subdir which holds the repetitions of the exp
        flags_hash = hash_dict(flags)
        flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

        for j in range(args.num_seeds_per_hparam):
            seed = init_seeds[j]
            cmd = generate_base_command(experiments.example_exp.run_example_exp, flags=dict(**flags, **{'seed': seed}))
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode='euler', promt=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_fun', type=str, default='quadratic')
    parser.add_argument('--method', type=str, default='random_search')
    parser.add_argument('--num_samples', type=int, default=100)

    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')
    parser.add_argument('--exp_name', type=str, required=True, default=None)
    parser.add_argument('--num_cpus', type=int, default=2, help='number of cpus to use')

    parser.add_argument('--num_hparam_samples', type=int, default=30)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=5)

    args = parser.parse_args()
    main(args)

# python experiments/launch_safe_bo_experiments.py --env CamelbackSinNoiseMetaEnv --opt_algo GooseUCB --model FPACOH --exp_name mar29 --num_hparam_samples 50 --num_seeds_per_hparam 20 --num_cpus 32

# python experiments/launch_safe_bo_experiments.py --env CamelbackSinNoiseMetaEnv --opt_algo GooseUCB --model Vanilla_GP --exp_name mar29 --num_hparam_samples 10 --num_seeds_per_hparam 20 --num_cpus 5