import json
import os
import pickle
import random
from pprint import pprint
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import wandb

from experiments.data_provider import _RACECAR_NOISE_STD_ENCODED
from experiments.online_rl_hardware.train_policy import ModelBasedRLConfig
from experiments.online_rl_hardware.train_policy import train_model_based_policy
from sim_transfer.models import BNN_FSVGD_SimPrior, BNN_FSVGD, BNN_SVGD
from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.simulators import AdditiveSim, PredictStateChangeWrapper, GaussianProcessSim
from sim_transfer.sims.simulators import RaceCarSim, StackedActionSimWrapper
from sim_transfer.sims.util import plot_rc_trajectory

WANDB_ENTITY = 'trevenl'
EULER_NAME = 'trevenl'
PRIORS = {'none_FVSGD',
          'none_SVGD',
          'high_fidelity',
          'low_fidelity',
          'high_fidelity_no_aditive_GP',
          }

""" LOAD REMOTE CONFIG """
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'remote_config.json'), 'r') as f:
    remote_config = json.load(f)
print('Remote config:')
pprint(remote_config)


def _get_random_hash() -> str:
    return "%032x" % random.getrandbits(128)


os.makedirs(remote_config['local_dir'], exist_ok=True)


def train_model_based_policy_remote(train_data: Dict,
                                    bnn_model: BatchedNeuralNetworkModel,
                                    run_remote: bool = True,
                                    verbosity: int = 1,
                                    **kwargs) -> Any:
    """ Trains a model-based policy on the remote machine and returns the trained model.
    Args:
        train_data: Dictionary containing the training data and potentially eval data
        It looks like: train_data = {
                                        'x_train': jnp.empty((0, state_dim + (1 + num_framestacks) * action_dim)),
                                        'y_train': jnp.empty((0, state_dim)),
                                    }
        Here y_train is the next state (not the difference)!

        run_remote: (bool) Whether to run the training on the remote machine or locally
        verbosity: (int) Verbosity level
        **kwargs: additional kwargs to pass to the train_model_based_policy function in train_policy.py

    Returns: the returned object/value of the train_model_based_policy function in train_policy.py
    """
    if not run_remote:
        # if not running remotely, just run the function locally and return the result
        return train_model_based_policy(train_data, bnn_model, **kwargs)

    # copy latest version of train_policy.py to remote and make sure remote directory exists
    os.system(f'scp {remote_config["local_script"]} {remote_config["remote_machine"]}:{remote_config["remote_script"]}')
    os.system(f'ssh {remote_config["remote_machine"]} "mkdir -p {remote_config["remote_dir"]}"')

    # dump train_data to local pkl file
    run_hash = _get_random_hash()
    train_data_path_local = os.path.join(remote_config['local_dir'], f'train_data_{run_hash}.pkl')
    with open(train_data_path_local, 'wb') as f:
        pickle.dump({'train_data': train_data, 'kwargs': kwargs}, f)
    if verbosity:
        print('[Local] Saved function input to', train_data_path_local)

    # transfer train_data + kwargs to remote
    train_data_path_remote = os.path.join(remote_config['remote_dir'], f'train_data_{run_hash}.pkl')
    os.system(f'scp {train_data_path_local} {remote_config["remote_machine"]}:{train_data_path_remote}')
    if verbosity:
        print('[Local] Transferring train data to remote')

    # run the train_policy.py script on the remote machine
    result_path_remote = os.path.join(remote_config['local_dir'], f'result_{run_hash}.pkl')
    command = f'{remote_config["remote_interpreter"]} {remote_config["remote_script"]} ' \
              f'--data_load_path {train_data_path_remote} --model_dump_path {result_path_remote}'
    if verbosity:
        print('[Local] Executing command:', command)
    os.system(f'ssh {remote_config["remote_machine"]} "{command}"')

    # transfer result back to local
    result_path_local = os.path.join(remote_config['local_dir'], f'result_{run_hash}.pkl')
    if verbosity:
        print('[Local] Transferring result back to local')
    os.system(f'scp {remote_config["remote_machine"]}:{result_path_remote} {result_path_local}')

    # load result
    with open(result_path_local, 'rb') as f:
        result = pickle.load(f)
    if verbosity:
        print('[Local] Loaded result from:', result_path_local)
    return result


class MainConfig(NamedTuple):
    horizon_len: int = 64
    seed: int = 0
    project_name: str = 'OnlineRL_RCCar'
    num_episodes: int = 20
    bnn_train_steps: int = 40_000
    sac_num_env_steps: int = 1_000_000
    learnable_likelihood_std: int = 1
    reset_bnn: int = 0
    sim_prior: str = 'none_SVGD'
    include_aleatoric_noise: int = 1
    best_bnn_model: int = 1
    best_policy: int = 1
    predict_difference: int = 1
    margin_factor: float = 20.0
    ctrl_cost_weight: float = 0.005
    num_stacked_actions: int = 3
    delay: float = 3 / 30
    max_replay_size_true_data_buffer: int = 10_000
    likelihood_exponent: float = 0.5
    data_batch_size: int = 32
    bandwidth_svgd: float = 0.2
    length_scale_aditive_sim_gp: float = 10.0
    num_f_samples: int = 512
    num_measurement_points: int = 16


def main(config: MainConfig = MainConfig(), run_remote: bool = False, encode_angle: bool = True, wandb_tag: str = ''):
    rng_key_env, rng_key_model, rng_key_rollouts = jax.random.split(jax.random.PRNGKey(config.seed), 3)

    env = RCCarSimEnv(encode_angle=encode_angle,
                      action_delay=config.delay,
                      use_tire_model=True,
                      use_obs_noise=True,
                      ctrl_cost_weight=config.ctrl_cost_weight,
                      margin_factor=config.margin_factor,
                      )

    # initialize train_data as empty arrays
    train_data = {
        'x_train': jnp.empty((0, env.dim_state[-1] + (1 + config.num_stacked_actions) * env.dim_action[-1])),
        'y_train': jnp.empty((0, env.dim_state[-1])),
    }

    ################################################################################
    #############################  Setup hyperparameters  ##########################
    ################################################################################
    """Setup key"""
    key = jr.PRNGKey(config.seed)
    key_sim, key_run_episodes = jr.split(key, 2)

    """Setup car reward kwargs"""
    car_reward_kwargs = dict(encode_angle=encode_angle,
                             ctrl_cost_weight=config.ctrl_cost_weight,
                             margin_factor=config.margin_factor)

    """Setup SAC config dict"""
    num_env_steps_between_updates = 16
    num_envs = 64
    sac_kwargs = dict(num_timesteps=config.sac_num_env_steps,
                      num_evals=20,
                      reward_scaling=10,
                      episode_length=config.horizon_len,
                      episode_length_eval=2 * config.horizon_len,
                      action_repeat=1,
                      discounting=0.99,
                      lr_policy=3e-4,
                      lr_alpha=3e-4,
                      lr_q=3e-4,
                      max_grad_norm=5.0,
                      num_envs=num_envs,
                      batch_size=64,
                      grad_updates_per_step=num_env_steps_between_updates * num_envs,
                      num_env_steps_between_updates=num_env_steps_between_updates,
                      tau=0.005,
                      wd_policy=0,
                      wd_q=0,
                      wd_alpha=0,
                      num_eval_envs=2 * num_envs,
                      max_replay_size=5 * 10 ** 4,
                      min_replay_size=2 ** 11,
                      policy_hidden_layer_sizes=(64, 64),
                      critic_hidden_layer_sizes=(64, 64),
                      normalize_observations=True,
                      deterministic_eval=True,
                      wandb_logging=True)

    total_config = sac_kwargs | config._asdict() | car_reward_kwargs
    wandb.init(
        entity=WANDB_ENTITY,
        dir='/cluster/scratch/' + EULER_NAME,
        project=config.project_name,
        config=total_config,
        tags=[] if wandb_tag == '' else [wandb_tag],
    )

    """ Setup BNN"""
    sim = RaceCarSim(encode_angle=True, use_blend=config.sim_prior == 'high_fidelity', car_id=2)
    if config.num_stacked_actions > 0:
        sim = StackedActionSimWrapper(sim, num_stacked_actions=config.num_stacked_actions, action_size=2)
    if config.predict_difference:
        sim = PredictStateChangeWrapper(sim)

    standard_params = {
        'input_size': sim.input_size,
        'output_size': sim.output_size,
        'rng_key': key_sim,
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_data': True,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': bool(config.learnable_likelihood_std),
        'likelihood_exponent': config.likelihood_exponent,
        'hidden_layer_sizes': [64, 64, 64],
        'normalization_stats': sim.normalization_stats,
        'data_batch_size': config.data_batch_size,
        'hidden_activation': jax.nn.leaky_relu,
        'num_train_steps': config.bnn_train_steps,

    }

    if config.sim_prior == 'none_FVSGD':
        bnn = BNN_FSVGD(
            **standard_params,
            domain=sim.domain,
            bandwidth_svgd=config.bandwidth_svgd,
        )
    elif config.sim_prior == 'none_SVGD':
        bnn = BNN_SVGD(
            **standard_params,
            bandwidth_svgd=1.0,
        )
    elif config.sim_prior == 'high_fidelity_no_aditive_GP':
        bnn = BNN_FSVGD_SimPrior(
            **standard_params,
            domain=sim.domain,
            function_sim=sim,
            score_estimator='gp',
            num_f_samples=config.num_f_samples,
            bandwidth_svgd=config.bandwidth_svgd,
            num_measurement_points=config.num_measurement_points,
        )
    else:
        if config.sim_prior == 'high_fidelity':
            outputscales_racecar = [0.008, 0.008, 0.009, 0.009, 0.05, 0.05, 0.20]
        elif config.sim_prior == 'low_fidelity':
            outputscales_racecar = [0.008, 0.008, 0.01, 0.01, 0.08, 0.08, 0.5]
        else:
            raise ValueError(f'Invalid sim prior: {config.sim_prior}')

        sim = AdditiveSim(base_sims=[sim,
                                     GaussianProcessSim(sim.input_size, sim.output_size,
                                                        output_scale=outputscales_racecar,
                                                        length_scale=config.length_scale_aditive_sim_gp,
                                                        consider_only_first_k_dims=None)
                                     ])

        bnn = BNN_FSVGD_SimPrior(
            **standard_params,
            domain=sim.domain,
            function_sim=sim,
            score_estimator='gp',
            num_f_samples=config.num_f_samples,
            bandwidth_svgd=config.bandwidth_svgd,
            num_measurement_points=config.num_measurement_points,
        )

    ################################################################################
    ################################################################################
    ################################################################################
    mbrl_config = ModelBasedRLConfig(
        x_dim=7,
        u_dim=2,
        num_stacked_actions=config.num_stacked_actions,
        max_replay_size_true_data_buffer=config.max_replay_size_true_data_buffer,
        include_aleatoric_noise=bool(config.include_aleatoric_noise),
        car_reward_kwargs=car_reward_kwargs,
        sac_kwargs=sac_kwargs,
        reset_bnn=bool(config.reset_bnn),
        return_best_bnn=bool(config.best_bnn_model),
        return_best_policy=bool(config.best_policy),
        predict_difference=bool(config.predict_difference),
        bnn_training_test_ratio=0.2,
        max_num_episodes=100)

    for episode_id in range(1, config.num_episodes + 1):
        print('\n\n ------- Episode', episode_id)
        key, key_episode = jr.split(key)

        # train model & policy
        policy, bnn = train_model_based_policy_remote(train_data=train_data,
                                                      bnn_model=bnn,
                                                      config=mbrl_config,
                                                      key=key_episode,
                                                      episode_idx=episode_id,
                                                      run_remote=run_remote)
        print(episode_id, policy)

        # perform policy rollout on the car
        stacked_actions = jnp.zeros(shape=(config.num_stacked_actions * mbrl_config.u_dim,))
        actions = []
        obs = env.reset()
        obs = jnp.concatenate([obs, stacked_actions])
        trajectory = [obs]
        rewards = []
        pure_obs = []
        for i in range(200):
            rng_key_rollouts, rng_key_act = jr.split(rng_key_rollouts)
            act = policy(obs)
            obs, reward, _, _ = env.step(act)
            rewards.append(reward)
            actions.append(act)
            pure_obs.append(obs)
            obs = jnp.concatenate([obs, stacked_actions])
            trajectory.append(obs)
            if config.num_stacked_actions > 0:
                stacked_actions = jnp.concatenate([stacked_actions[mbrl_config.u_dim:], act])

        trajectory = jnp.array(trajectory)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        pure_obs = jnp.array(pure_obs)

        fig, axes = plot_rc_trajectory(pure_obs,
                                       actions,
                                       encode_angle=encode_angle,
                                       show=False)
        wandb.log({'True_trajectory_path': wandb.Image(fig),
                   'reward_on_true_system': jnp.sum(rewards),
                   'x_axis/episode': episode_id})
        plt.close('all')

        # add observations and actions to train_data
        train_data['x_train'] = jnp.concatenate([train_data['x_train'],
                                                 jnp.concatenate([actions, trajectory[:-1]], axis=-1)], axis=0)
        train_data['y_train'] = jnp.concatenate([train_data['y_train'], trajectory[1:, :mbrl_config.x_dim]], axis=0)
        print(f'Size of train_data in episode {episode_id}:', train_data['x_train'].shape[0])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Meta-BO run')
    parser.add_argument('--seed', type=int, default=914)
    parser.add_argument('--prior', type=str, default='none_FVSGD')
    parser.add_argument('--project_name', type=str, default='OnlineRL_RCCar')
    parser.add_argument('--run_remote', type=int, default=0)
    parser.add_argument('--wandb_tag', type=str, default='')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    if not args.gpu:
        # disable gp
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'

    assert args.prior in PRIORS, f'Invalid prior: {args.prior}'
    main(config=MainConfig(sim_prior=args.prior,
                           seed=args.seed,
                           project_name=args.project_name, ),
         run_remote=bool(args.run_remote), )
