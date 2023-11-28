import json
import os
import pickle
import random
import sys
from pprint import pprint
from typing import Any, NamedTuple
from functools import cache

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import wandb

from experiments.online_rl_hardware.train_policy import ModelBasedRLConfig
from experiments.online_rl_hardware.train_policy import train_model_based_policy
from experiments.online_rl_hardware.utils import (set_up_bnn_dynamics_model, set_up_dummy_sac_trainer,
                                                  dump_trajectory_summary, execute,
                                                  prepare_init_transitions_for_car_env)
from experiments.util import Logger, RESULT_DIR

from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.util import plot_rc_trajectory

WANDB_ENTITY = 'jonasrothfuss'
EULER_ENTITY = 'rojonas'
WANDB_LOG_DIR_EULER = '/cluster/scratch/' + EULER_ENTITY
PRIORS = {'none_FVSGD',
          'none_SVGD',
          'high_fidelity',
          'low_fidelity',
          'high_fidelity_no_aditive_GP',
          }


@cache
def _load_remote_config(machine: str):
    # load remote config
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'remote_config.json'), 'r') as f:
        remote_config = json.load(f)

    # choose machine
    assert machine in remote_config, f'Machine {machine} not found in remote config. ' \
                                     f'Available machines: {list(remote_config.keys())}'
    remote_config = remote_config[machine]

    # create local director if it does not exist
    os.makedirs(remote_config['local_dir'], exist_ok=True)

    # print remote config
    print(f'Remote config [{machine}]:')
    pprint(remote_config)
    print('')

    return remote_config


def _get_random_hash() -> str:
    return "%032x" % random.getrandbits(128)


def train_model_based_policy_remote(*args,
                                    verbosity: int = 2,
                                    machine: str = 'euler',
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
    if machine == 'local':
        # if not running remotely, just run the function locally and return the result
        return train_model_based_policy(*args, **kwargs)
    rmt_cfg = _load_remote_config(machine=machine)

    # copy latest version of train_policy.py to remote and make sure remote directory exists
    execute(f'scp {rmt_cfg["local_script"]} {rmt_cfg["remote_machine"]}:{rmt_cfg["remote_script"]}', verbosity)
    execute(f'ssh {rmt_cfg["remote_machine"]} "mkdir -p {rmt_cfg["remote_dir"]}"', verbosity)

    # dump train_data to local pkl file
    run_hash = _get_random_hash()
    train_data_path_local = os.path.join(rmt_cfg['local_dir'], f'train_data_{run_hash}.pkl')
    with open(train_data_path_local, 'wb') as f:
        pickle.dump({'args': args, 'kwargs': kwargs}, f)
    if verbosity:
        print('[Local] Saved function input to', train_data_path_local)

    # transfer train_data + kwargs to remote
    train_data_path_remote = os.path.join(rmt_cfg['remote_dir'], f'train_data_{run_hash}.pkl')
    execute(f'scp {train_data_path_local} {rmt_cfg["remote_machine"]}:{train_data_path_remote}', verbosity)
    if verbosity:
        print('[Local] Transferring train data to remote')

    # run the train_policy.py script on the remote machine
    result_path_remote = os.path.join(rmt_cfg['remote_dir'], f'result_{run_hash}.pkl')
    command = f'export PYTHONPATH={rmt_cfg["remote_pythonpath"]} && ' \
              f'{rmt_cfg["remote_interpreter"]} {rmt_cfg["remote_script"]} ' \
              f'--data_load_path {train_data_path_remote} --model_dump_path {result_path_remote}'
    if verbosity:
        print('[Local] Executing command:', command)
    execute(f'ssh {rmt_cfg["remote_machine"]} "{command}"', verbosity)

    # transfer result back to local
    result_path_local = os.path.join(rmt_cfg['local_dir'], f'result_{run_hash}.pkl')
    if verbosity:
        print('[Local] Transferring result back to local')
    execute(f'scp {rmt_cfg["remote_machine"]}:{result_path_remote} {result_path_local}', verbosity)

    # load result
    with open(result_path_local, 'rb') as f:
        result = pickle.load(f)
    if verbosity:
        print('[Local] Loaded result from:', result_path_local)
    return result


class MainConfig(NamedTuple):
    num_env_steps: int = 200
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
    deterministic_policy: int = 1
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
    initial_state_fraction: float = 0.5
    sim: int = 1
    control_time_ms: float = 24.


def main(config: MainConfig = MainConfig(), encode_angle: bool = True,
         machine: str = 'local'):
    rng_key_env, rng_key_model, rng_key_rollouts = jax.random.split(jax.random.PRNGKey(config.seed), 3)

    """Setup car reward kwargs"""
    car_reward_kwargs = dict(encode_angle=encode_angle,
                             ctrl_cost_weight=config.ctrl_cost_weight,
                             margin_factor=config.margin_factor)
    """Set up env"""
    if bool(config.sim):
        env = RCCarSimEnv(encode_angle=encode_angle,
                          action_delay=config.delay,
                          use_tire_model=True,
                          use_obs_noise=True,
                          ctrl_cost_weight=config.ctrl_cost_weight,
                          margin_factor=config.margin_factor,
                          )
    else:
        from sim_transfer.hardware.car_env import CarEnv
        # We do not perform frame stacking in the env and do it manually here in the rollout function.
        env = CarEnv(
            encode_angle=encode_angle,
            car_id=2,
            control_time_ms=config.control_time_ms,
            max_throttle=0.4,
            car_reward_kwargs=car_reward_kwargs,
            num_frame_stacks=0

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
    key_bnn, key_run_episodes, key_dummy_sac_trainer, key = jr.split(key, 4)

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

    """ WANDB & Logging configuration """
    wandb_config = {'project': config.project_name, 'entity': WANDB_ENTITY, 'resume': 'allow',
                    'dir': WANDB_LOG_DIR_EULER if os.path.isdir(WANDB_LOG_DIR_EULER) else '/tmp/',
                    'config': total_config, 'settings': {'_service_wait': 300}}
    wandb.init(**wandb_config)
    wandb_config['id'] = wandb.run.id
    remote_training = not (machine == 'local')

    dump_dir = os.path.join(RESULT_DIR, 'online_rl_hardware', wandb_config['id'])
    os.makedirs(dump_dir, exist_ok=True)
    log_path = os.path.join(dump_dir, f"{wandb_config['id']}.log")

    if machine == 'euler':
        wandb_config_remote = wandb_config | {'dir': '/cluster/scratch/' + EULER_ENTITY}
    else:
        wandb_config_remote = wandb_config | {'dir': '/tmp/'}

    if remote_training:
        wandb.finish()
    sys.stdout = Logger(log_path, stream=sys.stdout)
    sys.stderr = Logger(log_path, stream=sys.stderr)
    print(f'\nDumping trajectories and logs to {dump_dir}\n')

    """ Setup BNN """
    bnn = set_up_bnn_dynamics_model(config=config, key=key_bnn)

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

    initial_states_fraction = max(min(config.initial_state_fraction, 0.9999), 0.0)
    init_state_points = lambda true_buffer_points: int(initial_states_fraction * true_buffer_points
                                                       / (1 - initial_states_fraction))

    """ Set up dummy SAC trainer for getting the policy from policy params """
    dummy_sac_trainer = set_up_dummy_sac_trainer(main_config=config, mbrl_config=mbrl_config, key=key)

    """ Main loop over episodes """
    for episode_id in range(1, config.num_episodes + 1):

        if remote_training:
            wandb.init(**wandb_config)
        sys.stdout = Logger(log_path, stream=sys.stdout)
        sys.stderr = Logger(log_path, stream=sys.stderr)
        print('\n\n------- Episode', episode_id)

        key, key_episode = jr.split(key)
        key_episode, key_init_buffer = jr.split(key_episode)

        num_points = train_data['x_train'].shape[0]
        num_init_state_points = init_state_points(num_points)
        if num_init_state_points > 0:
            init_transitions = prepare_init_transitions_for_car_env(key=key_init_buffer,
                                                                    number_of_samples=num_init_state_points,
                                                                    num_frame_stack=config.num_stacked_actions)
        else:
            init_transitions = None
        # train model & policy
        policy_params, bnn = train_model_based_policy_remote(
            train_data=train_data, bnn_model=bnn, config=mbrl_config, key=key_episode,
            episode_idx=episode_id, machine=machine, wandb_config=wandb_config_remote,
            remote_training=remote_training, reset_buffer_transitions=init_transitions)

        # get  allable policy from policy params
        def policy(x, key: jr.PRNGKey = jr.PRNGKey(0)):
            return dummy_sac_trainer.make_policy(policy_params,
                                                 deterministic=bool(config.deterministic_policy))(x, key)[0]

        # perform policy rollout on the car
        stacked_actions = jnp.zeros(shape=(config.num_stacked_actions * mbrl_config.u_dim,))
        obs = jnp.concatenate([env.reset(), stacked_actions])
        trajectory = [obs]
        actions, rewards, pure_obs = [], [], []
        for i in range(config.num_env_steps):
            rng_key_rollouts, rng_key_act = jr.split(rng_key_rollouts)
            act = policy(obs, rng_key_act)
            obs, reward, _, _ = env.step(act)
            rewards.append(reward)
            actions.append(act)
            pure_obs.append(obs)
            obs = jnp.concatenate([obs, stacked_actions])
            trajectory.append(obs)
            if config.num_stacked_actions > 0:
                stacked_actions = jnp.concatenate([stacked_actions[mbrl_config.u_dim:], act])

        # logging and saving
        trajectory, actions, rewards, pure_obs = map(lambda arr: jnp.array(arr),
                                                     [trajectory, actions, rewards, pure_obs])

        traj_summary = {'episode_id': episode_id, 'trajectory': trajectory, 'actions': actions, 'rewards': rewards,
                        'obs': pure_obs, 'return': jnp.sum(rewards)}
        dump_trajectory_summary(dump_dir=dump_dir, episode_id=episode_id, traj_summary=traj_summary, verbosity=1)

        fig, axes = plot_rc_trajectory(pure_obs, actions, encode_angle=encode_angle, show=False)
        wandb.log({'True_trajectory_path': wandb.Image(fig),
                   'reward_on_true_system': jnp.sum(rewards),
                   'x_axis/episode': episode_id})
        plt.close('all')

        # add observations and actions to train_data
        train_data['x_train'] = jnp.concatenate([train_data['x_train'],
                                                 jnp.concatenate([trajectory[:-1], actions], axis=-1)], axis=0)
        train_data['y_train'] = jnp.concatenate([train_data['y_train'], trajectory[1:, :mbrl_config.x_dim]], axis=0)
        print(f'Size of train_data in episode {episode_id}:', train_data['x_train'].shape[0])

        if remote_training:
            wandb.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Meta-BO run')
    parser.add_argument('--seed', type=int, default=914)
    parser.add_argument('--project_name', type=str, default='OnlineRL_RCCar')
    parser.add_argument('--machine', type=str, default='optimality')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--sim', type=int, default=1)
    parser.add_argument('--control_time_ms', type=float, default=24.)

    parser.add_argument('--prior', type=str, default='none_FVSGD')
    parser.add_argument('--num_env_steps', type=int, default=200, info='number of steps in the environment per episode')
    parser.add_argument('--reset_bnn', type=int, default=0)
    parser.add_argument('--deterministic_policy', type=int, default=1)
    parser.add_argument('--initial_state_fraction', type=float, default=0.5)
    args = parser.parse_args()

    if not args.gpu:
        # disable gp
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    assert args.prior in PRIORS, f'Invalid prior: {args.prior}'
    main(config=MainConfig(sim_prior=args.prior,
                           seed=args.seed,
                           project_name=args.project_name,
                           num_env_steps=args.num_env_steps,
                           reset_bnn=args.reset_bnn,
                           sim=args.sim,
                           control_time_ms=args.control_time_ms,
                           deterministic_policy=args.control_time_ms,
                           initial_state_fraction=args.initial_state_fraction,
                           ),
         machine=args.machine)
