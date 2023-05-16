from baselines.pacoh_nn.pacoh_nn_regression import PACOH_NN_Regression
from experiments.data_provider import provide_data_and_sim
from experiments.util import hash_dict, NumpyArrayEncoder

import os
import tensorflow as tf
import numpy as np
import jax
import json
import time
import datetime
import argparse

def meta_learning_experiment(
        # data parameters
        model: str,
        data_source: str,
        num_samples_train: int,
        data_seed: int = 981648,
        num_tasks: int = 200,
        model_seed: int = 892616,
        likelihood_std: float = 0.1,
        num_iter_meta_train: int = 20000,
        prior_weight: float = 1.0,
        meta_batch_size: int = 16,
        batch_size: int = 32,
        bandwidth: float = 10.
):
    # provide data and sim
    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
        data_source=data_source,
        data_spec={'num_samples_train': num_samples_train},
        data_seed=data_seed)

    meta_test_data = [tuple(map(lambda arr: np.array(arr), [x_train, y_train, x_test, y_test]))]

    # generate meta-training data from sim
    sim_data_key = jax.random.PRNGKey(model_seed + 1)
    meta_training_data = []
    for key in jax.random.split(sim_data_key, num_tasks):
        key_x, key_y = jax.random.split(key)
        x = sim.domain.sample_uniformly(key=key_x, sample_shape=500)
        y = sim.sample_function_vals(x=x, num_samples=1, rng_key=key_y)[0]
        meta_training_data.append((np.array(x), np.array(y)))

    if model == 'PACOH':
        # run meta-learning
        pacoh_model = PACOH_NN_Regression(meta_training_data, random_seed=model_seed,
                                          num_iter_meta_train=num_iter_meta_train,
                                      num_iter_meta_test=5000,
                                      learn_likelihood=False,
                                      hidden_layer_sizes=(64, 64, 64),
                                      activation='leaky_relu',
                                      likelihood_std=likelihood_std,
                                      prior_weight=prior_weight,
                                          meta_batch_size=meta_batch_size,
                                          batch_size=batch_size,
                                          bandwidth=bandwidth)

        # run meta-testing
        pacoh_model.meta_fit(meta_val_data=meta_test_data, eval_period=5000)

        y_preds, pred_dist = pacoh_model.meta_predict(x_train, y_train, x_test)
        nll = - float(tf.reduce_mean(pred_dist.log_prob(y_test)))
        rmse = float(tf.sqrt(tf.reduce_mean(tf.reduce_sum((pred_dist.mean() - y_test)**2, axis=-1))))
        avg_std = float(tf.reduce_mean(pred_dist.stddev()))
        eval_stats = {'nll': nll, 'rmse': rmse, 'avg_std': avg_std}
    else:
        raise NotImplementedError(f'Unknown model {model}')
    return eval_stats


def main(args):
    ''' generate experiment hash and set up redirect of output streams '''
    exp_params = args.__dict__
    exp_result_folder = exp_params.pop('exp_result_folder')
    use_wandb = exp_params.pop('use_wandb')
    exp_name = exp_params.pop('exp_name')
    exp_hash = hash_dict(exp_params)

    if exp_result_folder is not None:
        os.makedirs(exp_result_folder, exist_ok=True)

    from pprint import pprint
    print('\nExperiment parameters:')
    pprint(exp_params)
    print('')

    """ Experiment core """
    t_start = time.time()

    eval_metrics = meta_learning_experiment(**exp_params)
    t_end = time.time()

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


if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # general args
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--use_wandb', type=bool, default=False)

    # data parameters
    parser.add_argument('--data_source', type=str, default='pendulum')
    parser.add_argument('--num_samples_train', type=int, default=20)
    parser.add_argument('--data_seed', type=int, default=77698)
    parser.add_argument('--num_tasks', type=int, default=200)

    # model parameters
    parser.add_argument('--model_seed', type=int, default=892616)
    parser.add_argument('--likelihood_std', type=float, default=0.1)
    parser.add_argument('--num_iter_meta_train', type=int, default=20000)
    parser.add_argument('--prior_weight', type=float, default=1.0)
    parser.add_argument('--meta_batch_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bandwidth', type=float, default=10.)

    parser.add_argument('--model', type=str, default='PACOH')

    args = parser.parse_args()
    main(args)
