import time
import json
import os
import argparse
import jax
import jax.numpy as jnp
import sys
import copy
import datetime
import wandb

from typing import List, Union
from experiments.util import hash_dict, NumpyArrayEncoder
from experiments.data_provider import provide_data_and_sim, DATASET_CONFIGS
from sim_transfer.models import (BNN_SVGD, BNN_FSVGD, BNN_FSVGD_SimPrior, BNN_MMD_SimPrior, BNN_SVGD_DistillPrior,
                                 BNNGreyBox)

from sim_transfer.sims.simulators import AdditiveSim, GaussianProcessSim, PredictStateChangeWrapper

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

ACTIVATION_DICT = {
    'relu': jax.nn.relu,
    'leaky_relu': jax.nn.leaky_relu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid,
    'elu': jax.nn.elu,
    'softplus': jax.nn.softplus,
    'swish': jax.nn.swish,
}

OUTPUTSCALES_RCCAR = [0.008, 0.008, 0.03, 0.03, 0.3, 0.3, 1.5]
OUTPUTSCALES_RCCAR_HF = [0.008, 0.008, 0.01, 0.01, 0.08, 0.08, 0.5]


def regression_experiment(
        # data parameters
        data_source: str,
        num_samples_train: int,
        data_seed: int = 981648,
        pred_diff: bool = False,

        # logging parameters
        use_wandb: bool = False,

        # standard BNN parameters
        model: str = 'BNN_SVGD',
        model_seed: int = 892616,
        likelihood_std: Union[List[float], float] = 0.1,
        data_batch_size: int = 8,
        min_train_steps: int = 2500,
        num_epochs: int = 60,
        max_train_steps: int = 100_000,
        num_sim_model_train_steps: int = 5_000,
        lr: float = 1e-3,
        hidden_activation: str = 'leaky_relu',
        num_layers: int = 3,
        layer_size: int = 64,
        normalize_likelihood_std: bool = False,
        learn_likelihood_std: bool = False,
        likelihood_exponent: float = 1.0,
        likelihood_reg: float = 0.0,
        # SVGD parameters
        num_particles: int = 20,
        bandwidth_svgd: float = 10.0,
        weight_prior_std: float = 0.5,
        bias_prior_std: float = 1e1,

        # FSVGD parameters
        bandwidth_gp_prior: float = 0.4,
        num_measurement_points: int = 32,

        # FSVGD_Sim_Prior parameters
        bandwidth_score_estim: float = None,
        ssge_kernel_type: str = 'IMQ',
        num_f_samples: int = 128,

        switch_score_estimator_frac: float = 0.75,
        added_gp_lengthscale: float = 5.,
        added_gp_outputscale: Union[List[float], float] = 0.05,

        # BNN_SVGD_DistillPrior
        num_distill_steps: int = 500000,
):
    num_train_steps = min(num_epochs * num_samples_train // data_batch_size + min_train_steps, max_train_steps)
    # provide data and sim
    if 'hf' in model:
        use_hf_sim = True
    else:
        use_hf_sim = False
    if 'real_racecar' in data_source:
        x_train, y_train, x_test, y_test, sim_lf = provide_data_and_sim(
            data_source=data_source,
            data_spec={'num_samples_train': num_samples_train, 'sampling': 'iid',
                       'use_hf_sim': use_hf_sim, 'num_samples_test': 6000,
                       'num_stacked_actions': 3},
            data_seed=data_seed)
        x_train = x_train[:num_samples_train]
        y_train = y_train[:num_samples_train]
    else:

        x_train, y_train, x_test, y_test, sim_lf = provide_data_and_sim(
            data_source=data_source,
            data_spec={'num_samples_train': num_samples_train},
            data_seed=data_seed)

    # handle pred diff mode
    if pred_diff:
        assert x_train.shape[-1] == sim_lf.input_size and y_train.shape[-1] == sim_lf.output_size
        y_train = y_train - x_train[..., :sim_lf.output_size]
        y_test = y_test - x_test[..., :sim_lf.output_size]
        sim_lf = PredictStateChangeWrapper(sim_lf)

    if model.endswith('_no_add_gp'):
        no_added_gp = True
        model = model.replace('_no_add_gp', '')
        added_gp_outputscale = 0.
    elif model in ['GreyBox', 'SysID', 'GreyBox_hf', 'SysID_hf']:
        no_added_gp = True
    else:
        no_added_gp = False

    # create additive sim with a GP on top of the sim prior to model the fidelity gap
    if no_added_gp:
        sim = sim_lf
    else:
        sim = AdditiveSim(base_sims=[sim_lf,
                                     GaussianProcessSim(
                                         sim_lf.input_size,
                                         sim_lf.output_size,
                                         output_scale=added_gp_outputscale,
                                         length_scale=added_gp_lengthscale)])

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
        'hidden_layer_sizes': [layer_size] * num_layers,
        'normalize_likelihood_std': normalize_likelihood_std,
        'learn_likelihood_std': bool(learn_likelihood_std),
        'likelihood_exponent': likelihood_exponent,
    }

    if model == 'BNN_SVGD':
        model = BNN_SVGD(num_particles=num_particles,
                         bandwidth_svgd=bandwidth_svgd,
                         weight_prior_std=weight_prior_std,
                         bias_prior_std=bias_prior_std,
                         likelihood_reg=likelihood_reg,
                         **standard_model_params)
    elif model == 'BNN_FSVGD':
        model = BNN_FSVGD(domain=sim.domain,
                          num_particles=num_particles,
                          bandwidth_svgd=bandwidth_svgd,
                          bandwidth_gp_prior=bandwidth_gp_prior,
                          likelihood_reg=likelihood_reg,
                          num_measurement_points=num_measurement_points,
                          **standard_model_params)
    elif 'BNN_FSVGD_SimPrior' in model:
        score_estimator = model.split('_')[-1]
        assert score_estimator in ['SSGE', 'ssge', 'GP', 'gp', 'KDE', 'kde', 'nu-method', 'gp+nu-method']
        model = BNN_FSVGD_SimPrior(domain=sim.domain,
                                   function_sim=sim,
                                   num_particles=num_particles,
                                   bandwidth_svgd=bandwidth_svgd,
                                   num_measurement_points=num_measurement_points,
                                   bandwidth_score_estim=bandwidth_score_estim,
                                   ssge_kernel_type=ssge_kernel_type,
                                   num_f_samples=num_f_samples,
                                   score_estimator=score_estimator,
                                   switch_score_estimator_frac=switch_score_estimator_frac,
                                   **standard_model_params)
    elif model in ['GreyBox', 'SysID', 'GreyBox_hf', 'SysID_hf']:
        base_bnn = BNN_FSVGD(domain=sim.domain,
                             num_particles=num_particles,
                             bandwidth_svgd=bandwidth_svgd,
                             bandwidth_gp_prior=bandwidth_gp_prior,
                             likelihood_reg=likelihood_reg,
                             num_measurement_points=num_measurement_points,
                             **standard_model_params)
        model = BNNGreyBox(
            base_bnn=base_bnn,
            sim=sim,
            use_base_bnn=(model == 'GreyBox'),
            num_sim_model_train_steps=num_sim_model_train_steps,
        )
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
    model.fit_with_scan(x_train, y_train, x_test, y_test, log_to_wandb=use_wandb, log_period=1000)
    # eval model
    eval_metrics = model.eval(x_test, y_test, per_dim_metrics=True)
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
    if 'added_gp_outputscale' in exp_params:
        factor = 1
        if exp_params['added_gp_outputscale'] > 0:
            factor = exp_params['added_gp_outputscale']
        if 'racecar' in exp_params['data_source']:
            if 'hf' in exp_params['model']:
                outputscales = OUTPUTSCALES_RCCAR_HF
            else:
                outputscales = OUTPUTSCALES_RCCAR
            outputscales_racecar = factor * jnp.array(outputscales)
            if 'no_angvel' in exp_params['data_source']:
                outputscales_racecar = outputscales_racecar[:-1]
            elif 'only_pose' in exp_params['data_source']:
                outputscales_racecar = outputscales_racecar[:-3]
            exp_params['added_gp_outputscale'] = outputscales_racecar.tolist()
            # print(f"Setting added_gp_outputscale to data_source default value from DATASET_CONFIGS "
            #      f"which is {exp_params['added_gp_outputscale']}")
        elif 'pendulum' in exp_params['data_source']:
            exp_params['added_gp_outputscale'] = [factor * 0.05, 0.05, 0.5]
        elif 'Sergio' in exp_params['data_source']:
            from experiments.data_provider import DEFAULTS_SERGIO
            exp_params['added_gp_outputscale'] = [factor * 0.1 for _ in range(DEFAULTS_SERGIO['sergio_dim'])]
        elif 'Greenhouse' in exp_params['data_source']:
            exp_params['added_gp_outputscale'] = [factor * 0.1 for _ in range(16)]
        else:
            raise AssertionError('passed negative value for added_gp_outputscale')
    # set likelihood_std to default value if not specified
    if exp_params['likelihood_std'] is None:
        likelihood_std = DATASET_CONFIGS[args.data_source]['likelihood_std']['value']
        if 'no_angvel' in exp_params['data_source']:
            likelihood_std = likelihood_std[:-1]
        elif 'only_pose' in exp_params['data_source']:
            likelihood_std = likelihood_std[:-3]
        exp_params['likelihood_std'] = likelihood_std
        # print(f"Setting likelihood_std to data_source default value from DATASET_CONFIGS "
        #      f"which is {exp_params['likelihood_std']}")



    from pprint import pprint
    if not 'Sergio' in exp_params['data_source']:
        print('\nExperiment parameters:')
        pprint(exp_params)
        print('')

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
        exp_result_file = os.path.join(exp_result_folder, f'{exp_hash}.json')
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print(f'Dumped results to {exp_result_file}')

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
    parser.add_argument('--data_source', type=str, default='Greenhouse_hf')
    parser.add_argument('--pred_diff', type=int, default=1)
    parser.add_argument('--num_samples_train', type=int, default=5000)
    parser.add_argument('--data_seed', type=int, default=77698)

    # standard BNN parameters
    parser.add_argument('--model', type=str, default='BNN_FSVGD')
    parser.add_argument('--model_seed', type=int, default=892616)
    parser.add_argument('--likelihood_std', type=float, default=None)
    parser.add_argument('--learn_likelihood_std', type=int, default=0)
    parser.add_argument('--likelihood_reg', type=float, default=0.0)
    parser.add_argument('--data_batch_size', type=int, default=8)
    parser.add_argument('--min_train_steps', type=int, default=2500)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--max_train_steps', type=int, default=100_000)
    parser.add_argument('--num_sim_model_train_steps', type=int, default=5_000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_activation', type=str, default='leaky_relu')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--layer_size', type=int, default=64)
    parser.add_argument('--normalize_likelihood_std', type=bool, default=True)
    parser.add_argument('--likelihood_exponent', type=float, default=1.0)

    # SVGD parameters
    parser.add_argument('--num_particles', type=int, default=20)
    parser.add_argument('--bandwidth_svgd', type=float, default=10.0)
    parser.add_argument('--weight_prior_std', type=float, default=0.5)
    parser.add_argument('--bias_prior_std', type=float, default=1.0)

    # FSVGD parameters
    parser.add_argument('--bandwidth_gp_prior', type=float, default=0.4)
    parser.add_argument('--num_measurement_points', type=int, default=32)

    # FSVGD_SimPrior parameters
    parser.add_argument('--bandwidth_score_estim', type=float, default=None)
    parser.add_argument('--ssge_kernel_type', type=str, default='IMQ')
    parser.add_argument('--num_f_samples', type=int, default=128)
    parser.add_argument('--switch_score_estimator_frac', type=float, default=0.75)

    # Additive SimPrior GP parameters
    parser.add_argument('--added_gp_lengthscale', type=float, default=5.)
    parser.add_argument('--added_gp_outputscale', type=float, default=1.0)

    # FSVGD_SimPrior parameters
    parser.add_argument('--num_distill_steps', type=int, default=50000)

    args = parser.parse_args()
    main(args)
