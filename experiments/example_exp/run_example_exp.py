import time
import json
import os
import sys
import argparse

import numpy as np
from experiments.util import Logger, hash_dict, NumpyArrayEncoder


def global_random_search_minimize(target_fun: callable, uniform: bool, loc: float, scale: float,
                                  num_samples: int, rds: np.random.RandomState):
    if uniform:
        sample = rds.uniform(loc - scale, args.loc + scale, size=num_samples)
    else:
        sample = rds.normal(loc, scale, size=num_samples)

    fs = target_fun(sample)
    best_idx = np.argmin(fs)
    return sample[best_idx], fs[best_idx]

def greedy_hillclimb_minimize(target_fun: callable, pertubation_std: float, anneal_factor: float,
                              num_samples: int, rds: np.random.RandomState):
    assert 0.0 < anneal_factor <= 1.0
    x_best = rds.uniform(-10, 10)
    f_best = target_fun(x_best)

    for i in range(num_samples):
        eps_sample = rds.normal(0., pertubation_std)
        x_proposal = x_best + eps_sample
        f_proposal = target_fun(x_proposal)
        if f_proposal < f_best:
            x_best = x_proposal
            f_best = f_proposal
        pertubation_std*= anneal_factor

    return x_best, f_best


def main(args):
    """"""

    ''' generate experiment hash and set up redirect of output streams '''
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' %exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    """ Experiment core """
    t_start = time.time()
    rds = np.random.RandomState(args.seed)

    # select target function
    if args.target_fun == 'quadratic':
        fun = lambda x: (x - 4)**2
        f_min = 0.
    elif args.target_fun == 'bell':
        fun = lambda x: - np.exp(-(x-4)**2)
        f_min = -1.
    else:
        raise NotImplementedError

    # the actual experiment logic happens here
    # I just run some super simple random optimization methods on a target function and evaluate the results

    if args.method == 'random_search':
        x_best, f_best = global_random_search_minimize(fun, uniform=args.uniform, loc=args.loc, scale=args.scale,
                                                       num_samples=args.num_samples, rds=rds)
    elif args.method == 'hill_search':
        x_best, f_best = greedy_hillclimb_minimize(fun, pertubation_std=args.pertubation_std,
                                                   anneal_factor=args.anneal_factor, num_samples=args.num_samples,
                                                   rds=rds)
    else:
        raise NotImplementedError

    # compute some metrics
    x_diff = np.abs(x_best - 4.)
    f_diff = np.abs(f_best - f_min)
    eval_metrics = {
        'x_diff': x_diff,
        'f_diff': f_diff
    }

    t_end = time.time()


    """ Save experiment results and configuration """
    results_dict = {
        'evals': eval_metrics,
        'params': args.__dict__,
        'duration_total': t_end - t_start
    }

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        exp_result_file = os.path.join(args.exp_result_folder, '%s.json'%exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s'%exp_result_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # general args
    parser.add_argument('--target_fun', type=str, default='quadratic')
    parser.add_argument('--method', type=str, default='hill_search')
    parser.add_argument('--num_samples', type=int, default=100)

    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')

    # method related args
    # 1) random search
    parser.add_argument('--loc', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=5.0)
    parser.add_argument('--uniform', action='store_true', help='whether to use the uniform dist instead of normal')

    # 2) random hillclimb
    parser.add_argument('--pertubation_std', type=float, default=1.0)
    parser.add_argument('--anneal_factor', type=float, default=1.0)

    args = parser.parse_args()
    main(args)
