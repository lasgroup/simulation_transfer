import argparse
import copy
import pickle
from typing import Any, Dict, NamedTuple

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import wandb
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.rl.model_based_rl.learned_system import LearnedCarSystem
from sim_transfer.rl.model_based_rl.utils import split_data


def _load_data(data_load_path: str) -> Any:
    # loads the pkl file
    with open(data_load_path, 'rb') as f:
        data = pickle.load(f)
    return data


def _dump_model(model: Any, model_dump_path: str) -> None:
    # dumps the model in the model_dump_path
    with open(model_dump_path, 'wb') as f:
        pickle.dump(model, f)


class ModelBasedRLConfig(NamedTuple):
    x_dim: int
    u_dim: int
    num_stacked_actions: int
    max_replay_size_true_data_buffer: int = 10 ** 4
    include_aleatoric_noise: bool = True
    car_reward_kwargs: dict = None
    sac_kwargs: dict = None
    reset_bnn: bool = True
    return_best_bnn: bool = True
    return_best_policy: bool = True
    predict_difference: bool = True
    bnn_training_test_ratio: float = 0.2
    num_stacked_actions: int = 3
    max_num_episodes: int = 100


def train_model_based_policy(train_data: Dict,
                             bnn_model: BatchedNeuralNetworkModel,
                             key: chex.PRNGKey,
                             episode_idx: int,
                             config: ModelBasedRLConfig):
    """
    train_data = {'x_train': jnp.empty((0, state_dim + (1 + num_framestacks) * action_dim)),
                  'y_train': jnp.empty((0, state_dim))}
        Here y_train is the next state (not the difference)!!!!!
    """
    assert 'x_train' in train_data and 'y_train' in train_data
    x_all, y_all = train_data['x_train'], train_data['y_train']
    discounting = config.sac_kwargs['discounting']

    """Setup the data buffers"""

    dummy_obs = jnp.zeros(shape=(config.x_dim + config.u_dim * config.num_stacked_actions,))
    dummy_sample = Transition(observation=dummy_obs,
                              action=jnp.zeros(shape=(config.u_dim,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(0.99),
                              next_observation=dummy_obs)

    true_data_buffer = UniformSamplingQueue(
        max_replay_size=config.max_replay_size_true_data_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)

    """Insert the data into the buffers for the policy training"""
    num_training_points = x_all.shape[0]
    assert x_all.shape == (num_training_points, config.x_dim + config.u_dim * (config.num_stacked_actions + 1))
    assert y_all.shape == (num_training_points, config.x_dim)

    observations = x_all[:, :config.x_dim + config.num_stacked_actions * config.u_dim]
    actions = x_all[:, config.x_dim + config.num_stacked_actions * config.u_dim:]
    rewards = jnp.zeros(shape=(num_training_points,))
    discounts = jnp.ones(shape=(num_training_points,)) * discounting
    if config.num_stacked_actions > 0:
        old_stacked_actions = observations[:, config.x_dim:]
        next_stacked_actions = jnp.concatenate([old_stacked_actions[:, config.u_dim:], actions], axis=1)
        next_observations = jnp.concatenate([y_all, next_stacked_actions], axis=1)
    else:
        next_observations = y_all
    transitions = Transition(observation=observations,
                             action=actions,
                             reward=rewards,
                             discount=discounts,
                             next_observation=next_observations)

    key, key_init_buffer = jr.split(key)
    true_data_buffer_state = true_data_buffer.init(key_init_buffer)
    true_data_buffer_state = true_data_buffer.insert(true_data_buffer_state, transitions)

    """Train transition model"""
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
        bnn_model.fit(x_train=x_train, y_train=y_train, x_eval=x_test, y_eval=y_test, log_to_wandb=True,
                      keep_the_best=config.return_best_bnn, metrics_objective='eval_nll')

    """Train policy"""
    _sac_kwargs = config.sac_kwargs
    # TODO: Be careful!!
    if num_training_points == 0:
        print("We don't have any data for training the bnn model")
        _sac_kwargs = copy.deepcopy(_sac_kwargs)
        _sac_kwargs['num_timesteps'] = 10_000

    system = LearnedCarSystem(model=bnn_model,
                              include_noise=config.include_aleatoric_noise,
                              predict_difference=config.predict_difference,
                              num_frame_stack=config.num_stacked_actions,
                              **config.car_reward_kwargs)

    key_train, key_simulate, *keys_sys_params = jr.split(key, 4)
    env = BraxWrapper(system=system,
                      sample_buffer_state=true_data_buffer_state,
                      sample_buffer=true_data_buffer,
                      system_params=system.init_params(keys_sys_params[0]))

    # Here we create eval envs
    sac_trainer = SAC(environment=env,
                      eval_environment=env,
                      eval_key_fixed=True,
                      return_best_model=config.return_best_policy,
                      **_sac_kwargs, )

    params, metrics = sac_trainer.run_training(key=key_train)

    best_reward = np.max([summary['eval/episode_reward'] for summary in metrics])
    wandb.log({'best_trained_reward': best_reward,
               'x_axis/episode': episode_idx})

    make_inference_fn = sac_trainer.make_policy

    def policy(x):
        return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]

    return policy, bnn_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_load_path', type=str, required=True)
    parser.add_argument('--model_dump_path', type=str, required=True)
    args = parser.parse_args()

    # load data
    function_args = _load_data(args.data_load_path)
    train_data = function_args['train_data']
    kwargs = function_args['kwargs']
    print(f'[Remote] Executing train_model_based_policy function ... ')
    trained_model = train_model_based_policy(train_data, **kwargs)
    _dump_model(trained_model, args.model_dump_path)
    print(f'[Remote] Dumped trained model to {args.model_dump_path}')
