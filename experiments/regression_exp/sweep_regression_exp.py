from experiments.util import generate_run_commands, BASE_DIR

import wandb
import datetime
import argparse
import sys
import os

def main(args):
    sweep_config = {
        'command': [sys.executable, '-u', os.path.join(BASE_DIR, 'experiments/regression_exp/run_regression_exp.py'),
                    '${args}'],
        'method': 'random',
        'name': f'{args.exp_name}/{args.data_source}/{args.model}',
        'metric': {'goal': 'minimize',
                   'name': 'final_nll'},
        'parameters': {
            'data_source': {'value': args.data_source},
            'num_samples_train': {'value': args.num_samples_train},
            'model': {'value': args.model},
            'model_seed': {'values': [34985]},
            'bandwidth_svgd': {'distribution': 'log_uniform', 'min': -1., 'max': 3.},
            'num_train_steps': {'values': [20000, 30000]},
            'use_wandb': {'value': 1},
        },
        'run_cap': args.num_runs
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project='sim_transfer')

    num_nodes = args.num_runs if args.num_nodes < 0 else args.num_nodes
    generate_run_commands([f'module load eth_proxy && wandb agent sim_transfer/sim_transfer/{sweep_id}'] * num_nodes,
                          num_cpus=args.num_cpus,
                          mode=args.run_mode,
                          promt=True,
                          )



if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # sweep args
    parser.add_argument('--num_runs', type=int, default=32)
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--run_mode', type=str, default='euler')
    parser.add_argument('--num_nodes', type=int, default=-1)

    # general args
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--use_wandb', action='store_true', default=False)

    # data parameters
    parser.add_argument('--data_source', type=str, default='sinusoids1d')
    parser.add_argument('--num_samples_train', type=int, default=10)
    parser.add_argument('--data_seed', type=int, default=34985)



    # # standard BNN parameters
    parser.add_argument('--model', type=str, default='BNN_SVGD')
    # parser.add_argument('--model_seed', type=int, default=892616)
    # parser.add_argument('--likelihood_std', type=float, default=0.1)
    # parser.add_argument('--data_batch_size', type=int, default=16)
    # parser.add_argument('--num_train_steps', type=int, default=20000)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--hidden_activation', type=str, default='leaky_relu')
    # parser.add_argument('--num_layers', type=int, default=3)
    # parser.add_argument('--layer_size', type=int, default=64)
    #
    # # SVGD parameters
    # parser.add_argument('--num_particles', type=int, default=20)
    # parser.add_argument('--bandwidth_svgd', type=float, default=10.0)
    # parser.add_argument('--weight_prior_std', type=float, default=0.5)
    # parser.add_argument('--bias_prior_std', type=float, default=1.0)

    args = parser.parse_args()
    main(args)
