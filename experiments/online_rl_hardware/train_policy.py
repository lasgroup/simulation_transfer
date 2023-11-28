import argparse
import copy
import time
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import wandb
from brax.training.types import Transition
from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from typing import Dict

from sim_transfer.rl.model_based_rl.utils import split_data
from experiments.online_rl_hardware.utils import (load_data, dump_model, ModelBasedRLConfig, init_transition_buffer,
                                                  add_data_to_buffer, set_up_model_based_sac_trainer)


def train_model_based_policy(train_data: Dict,
                             bnn_model: BatchedNeuralNetworkModel,
                             key: chex.PRNGKey,
                             episode_idx: int,
                             config: ModelBasedRLConfig,
                             wandb_config: Dict,
                             remote_training: bool = False,
                             reset_buffer_transitions: Transition | None = None,
                             ):
    """
    train_data = {'x_train': jnp.empty((0, state_dim + (1 + num_framestacks) * action_dim)),
                  'y_train': jnp.empty((0, state_dim))}
        Here y_train is the next state (not the difference)!!!!!
    """
    assert 'x_train' in train_data and 'y_train' in train_data
    x_all, y_all = train_data['x_train'], train_data['y_train']
    num_training_points = x_all.shape[0]

    # reinitialize wandb
    if remote_training:
        wandb.init(**wandb_config)

    """ Setup the data buffers """
    key, key_buffer_init = jr.split(key, 2)
    true_data_buffer, true_data_buffer_state = init_transition_buffer(config=config, key=key_buffer_init)
    true_data_buffer, true_data_buffer_state = add_data_to_buffer(true_data_buffer, true_data_buffer_state,
                                                                  x_data=x_all, y_data=y_all, config=config)

    """Train transition model"""
    t = time.time()
    # Prepare data for training the transition model
    if num_training_points > 0:
        if config.predict_difference:
            y_all = y_all - x_all[:, :config.x_dim]
        key, key_split_data, key_reinit_model = jr.split(key, 3)
        x_train, x_test, y_train, y_test = split_data(x_all, y_all,
                                                      test_ratio=config.bnn_training_test_ratio,
                                                      key=key_split_data)

        # Train model
        if config.reset_bnn:
            bnn_model.reinit(rng_key=key_reinit_model)
        bnn_model.fit_with_scan(x_train=x_train, y_train=y_train, x_eval=x_test, y_eval=y_test, log_to_wandb=True,
                      keep_the_best=config.return_best_bnn, metrics_objective='eval_nll', log_period=2000)
    print(f'Time fo training the transition model: {time.time() - t:.2f} seconds')

    """Train policy"""
    t = time.time()
    if reset_buffer_transitions:
        sac_buffer_state = true_data_buffer.insert(true_data_buffer_state, reset_buffer_transitions)
    else:
        sac_buffer_state = true_data_buffer_state

    _sac_kwargs = config.sac_kwargs
    # TODO: Be careful!!
    if num_training_points == 0:
        print("We don't have any data for training the bnn model")
        _sac_kwargs = copy.deepcopy(_sac_kwargs)
        _sac_kwargs['num_timesteps'] = 10_000

    key, key_sac_training, key_sac_trainer_init = jr.split(key, 3)
    sac_trainer = set_up_model_based_sac_trainer(
        bnn_model=bnn_model, data_buffer=true_data_buffer, data_buffer_state=sac_buffer_state,
        key=key_sac_trainer_init, config=config, sac_kwargs=_sac_kwargs)

    policy_params, metrics = sac_trainer.run_training(key=key_sac_training)

    best_reward = np.max([summary['eval/episode_reward'] for summary in metrics])
    wandb.log({'best_trained_reward': best_reward, 'x_axis/episode': episode_idx})

    print(f'Time fo training the SAC policy: {time.time() - t:.2f} seconds')

    if remote_training:
        wandb.finish()

    return policy_params, bnn_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_load_path', type=str, required=True)
    parser.add_argument('--model_dump_path', type=str, required=True)
    args = parser.parse_args()

    # load data
    function_args = load_data(args.data_load_path)
    train_args = function_args['args']
    train_kwargs = function_args['kwargs']
    print(f'[Remote] Executing train_model_based_policy function ... ')
    trained_model = train_model_based_policy(*train_args, **train_kwargs)
    dump_model(trained_model, args.model_dump_path)
    print(f'[Remote] Dumped trained model to {args.model_dump_path}')
