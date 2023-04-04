import time
import json
import os
import argparse
import jax
import sys
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import datetime
import wandb
from typing import Dict, List, Tuple, Union
from experiments.util import Logger, hash_dict, NumpyArrayEncoder
from experiments.data_provider import provide_data_and_sim
from sim_transfer.models import BNN_SVGD, BNN_FSVGD, BNN_FSVGD_SimPrior, BNN_MMD_SimPrior, BNN_SVGD_DistillPrior

ACTIVATION_DICT = {
    'relu': jax.nn.relu,
    'leaky_relu': jax.nn.leaky_relu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid,
    'elu': jax.nn.elu,
    'softplus': jax.nn.softplus,
    'swish': jax.nn.swish,
}

def regression_experiment(
                          # data parameters
                          data_source: str,
                          num_samples_train: int,
                          data_seed: int = 981648,

                          # logging parameters
                          use_wandb: bool = False,

                          # standard BNN parameters
                          model: str = 'BNN_SVGD',
                          model_seed: int = 892616,
                          likelihood_std: float = 0.1,
                          data_batch_size: int = 8,
                          num_train_steps: int = 20000,
                          lr: float = 1e-3,
                          hidden_activation: str = 'leaky_relu',
                          num_layers: int = 3,
                          layer_size: int = 64,
                          normalize_likelihood_std: bool = False,

                          # SVGD parameters
                          num_particles: int = 20,
                          bandwidth_svgd: float = 10.0,
                          weight_prior_std: float = 0.5,
                          bias_prior_std: float = 1e1,

                          # FSVGD parameters
                          bandwidth_gp_prior: float = 0.4,
                          num_measurement_points: int = 32,

                          # FSVGD_Sim_Prior parameters
                          bandwidth_ssge: float = None,
                          bandwidth_kde: float = None,
                          num_f_samples: int = 100,

                          # BNN_SVGD_DistillPrior
                          num_distill_steps: int = 500000,
                          ):
    # provide data and sim
    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
        data_source=data_source,
        data_spec={'num_samples_train': num_samples_train},
        data_seed=data_seed)

    # setup standard model params
    standard_model_params = {
        'input_size': sim.input_size,
        'output_size': sim.output_size,
        'normalization_stats': sim.normalization_stats,
        'normalize_data': True,
        'rng_key': jax.random.PRNGKey(model_seed),
        'likelihood_std': likelihood_std,
        'data_batch_size': data_batch_size,
        'num_train_steps': num_train_steps,
        'lr': lr,
        'hidden_activation': ACTIVATION_DICT[hidden_activation],
        'hidden_layer_sizes': [layer_size]*num_layers,
        'normalize_likelihood_std': normalize_likelihood_std,
    }

    if model == 'BNN_SVGD':
        model = BNN_SVGD(num_particles=num_particles,
                         bandwidth_svgd=bandwidth_svgd,
                         weight_prior_std=weight_prior_std,
                         bias_prior_std=bias_prior_std,
                         **standard_model_params)
    elif model == 'BNN_FSVGD':
        model = BNN_FSVGD(domain=sim.domain,
                          num_particles=num_particles,
                          bandwidth_svgd=bandwidth_svgd,
                          bandwidth_gp_prior=bandwidth_gp_prior,
                          num_measurement_points=num_measurement_points,
                          **standard_model_params)
    elif 'BNN_FSVGD_SimPrior' in model:
        score_estimator = model.split('_')[-1]
        assert score_estimator in ['SSGE', 'ssge', 'GP', 'gp', 'KDE', 'kde']
        model = BNN_FSVGD_SimPrior(domain=sim.domain,
                                   function_sim=sim,
                                   num_particles=num_particles,
                                   bandwidth_svgd=bandwidth_svgd,
                                   num_measurement_points=num_measurement_points,
                                   bandwidth_ssge=bandwidth_ssge,
                                   bandwidth_kde=bandwidth_kde,
                                   num_f_samples=num_f_samples,
                                   score_estimator=score_estimator,
                                   **standard_model_params)
    elif model == 'BNN_MMD_SimPrior':
        model = BNN_MMD_SimPrior(domain=sim.domain,
                                 function_sim=sim,
                                 num_particles=num_particles,
                                 num_f_samples=num_f_samples,
                                 num_measurement_points=num_measurement_points,
                                 **standard_model_params)
    elif model == 'BNN_SVGD_DistillPrior':
        model = BNN_SVGD_DistillPrior(domain=sim.domain,
                                      function_sim=sim,
                                      num_particles=num_particles,
                                      bandwidth_svgd=bandwidth_svgd,
                                      num_measurement_points=num_measurement_points,
                                      num_f_samples=num_f_samples,
                                      num_distill_steps=num_distill_steps,
                                      **standard_model_params)

    else:
        raise NotImplementedError('Model {model} not implemented')

    # train model
    model.fit(x_train, y_train, x_test, y_test, log_to_wandb=use_wandb, log_period=1000)

    # eval model
    eval_metrics = model.eval(x_test, y_test)
    return eval_metrics


def main(args):
    """"""

    ''' generate experiment hash and set up redirect of output streams '''
    exp_params = args.__dict__
    exp_result_folder = exp_params.pop('exp_result_folder')
    use_wandb = exp_params.pop('use_wandb')
    exp_name = exp_params.pop('exp_name')
    exp_hash = hash_dict(exp_params)

    if exp_result_folder is not None:
        os.makedirs(exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(exp_result_folder, f'{exp_hash}.log ')
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    """ Experiment core """
    t_start = time.time()

    if use_wandb:
        # hash of experiments without seeds
        exp_params_no_seeds = copy.deepcopy(exp_params)
        [exp_params_no_seeds.pop(k) for k in ['model_seed', 'data_seed']]
        exp_hash_no_seeds = hash_dict(exp_params_no_seeds)

        wandb.init(project='sim_transfer', config=exp_params,
                   name=f'{exp_name}/{args.data_source}/{args.model}/{exp_hash}',
                   group=f'{exp_name}/{args.data_source}/{args.model}/{exp_hash_no_seeds}'
                   )

    eval_metrics = regression_experiment(**exp_params, use_wandb=use_wandb)

    t_end = time.time()

    if use_wandb:
        for key, val in eval_metrics.items():
            wandb.summary[key] = float(val)
        wandb.log({f'final_{key}': float(val) for key, val in eval_metrics.items()})

    """ Save experiment results and configuration """
    results_dict = {
        'evals': eval_metrics,
        'params': exp_params,
        'duration_total': t_end - t_start
    }

    if exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        exp_result_file = os.path.join(exp_result_folder, '%s.json'%exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s'%exp_result_file)

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # general args
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--use_wandb', type=bool, default=False)

    # data parameters
    parser.add_argument('--data_source', type=str, default='pendulum')
    parser.add_argument('--num_samples_train', type=int, default=10)
    parser.add_argument('--data_seed', type=int, default=77698)

    # standard BNN parameters
    parser.add_argument('--model', type=str, default='BNN_FSVGD_SimPrior_ssge')
    parser.add_argument('--model_seed', type=int, default=892616)
    parser.add_argument('--likelihood_std', type=float, default=0.1)
    parser.add_argument('--data_batch_size', type=int, default=8)
    parser.add_argument('--num_train_steps', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_activation', type=str, default='leaky_relu')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--layer_size', type=int, default=64)
    parser.add_argument('--normalize_likelihood_std', type=bool, default=True)

    # SVGD parameters
    parser.add_argument('--num_particles', type=int, default=20)
    parser.add_argument('--bandwidth_svgd', type=float, default=10.0)
    parser.add_argument('--weight_prior_std', type=float, default=0.5)
    parser.add_argument('--bias_prior_std', type=float, default=1.0)

    # FSVGD parameters
    parser.add_argument('--bandwidth_gp_prior', type=float, default=0.4)
    parser.add_argument('--num_measurement_points', type=int, default=16)

    # FSVGD_SimPrior parameters
    parser.add_argument('--bandwidth_ssge', type=float, default=None)
    parser.add_argument('--bandwidth_kde', type=float, default=None)
    parser.add_argument('--num_f_samples', type=int, default=64)

    # FSVGD_SimPrior parameters
    parser.add_argument('--num_distill_steps', type=int, default=50000)

    args = parser.parse_args()
    main(args)
