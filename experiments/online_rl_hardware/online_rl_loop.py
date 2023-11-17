import os

from experiments.online_rl_hardware.train_policy import train_model_based_policy
from sim_transfer.sims.envs import RCCarSimEnv
from typing import Any, Dict

import json
import jax
import pickle
import random
from pprint import pprint
import jax.numpy as jnp

""" LOAD REMOTE CONFIG """
with open('remote_config.json', 'r') as f:
    remote_config = json.load(f)
print('Remote config:')
pprint(remote_config)

def _get_random_hash() -> str:
    return "%032x"%random.getrandbits(128)

os.makedirs(remote_config['local_dir'], exist_ok=True)



def train_model_based_policy_remote(train_data: Dict, run_remote: bool = True, verbosity: int = 1, **kwargs) -> Any:
    """ Trains a model-based policy on the remote machine and returns the trained model.
    Args:
        train_data: Dictionary containing the training data and potentially eval data
        run_remote: (bool) Whether to run the training on the remote machine or locally
        verbosity: (int) Verbosity level
        **kwargs: additional kwargs to pass to the train_model_based_policy function in train_policy.py

    Returns: the returned object/value of the train_model_based_policy function in train_policy.py
    """
    if not run_remote:
        # if not running remotely, just run the function locally and return the result
        return train_model_based_policy(train_data, **kwargs)

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


def main(seed: int = 234238, num_episodes: int = 20, num_framestacks: int = 0):
    rng_key_env, rng_key_model, rng_key_rollouts = jax.random.split(jax.random.PRNGKey(seed), 3)
    env = RCCarSimEnv()

    # initialize train_data as empty arrays
    train_data = {
        'x_train': jnp.empty((0, env.dim_state[-1] + (1 + num_framestacks) * env.dim_action[-1])),
        'y_train': jnp.empty((0, env.dim_state[-1])),
    }

    for episode_id in range(1, num_episodes+1):
        print('\n\n ------- Episode', episode_id)

        # train model & policy
        # Note: sleep_time is only a random kwarg to test whether kwargs are passed correctly
        # TODO: remove sleep_time and rempalace with proper kwargs
        policy = train_model_based_policy_remote(train_data, sleep_time=2)
        print(episode_id, policy)

        # perform policy rollout on the car
        # TODO: implement the rollouts properly (i.e., add a proper policy and support for framestacks)

        actions = []
        trajectory = [env.reset()]
        for i in range(20):
            rng_key_rollouts, rng_key_act = jax.random.split(rng_key_rollouts)
            act = jax.random.uniform(rng_key_act, shape=(2,), minval=-1.0, maxval=1.0)  # takes random actions atm
            obs, reward, _, _ = env.step(act)
            actions.append(act)
            trajectory.append(obs)

        trajectory = jnp.array(trajectory)
        actions = jnp.array(actions)

        # add observations and actions to train_data
        train_data['x_train'] = jnp.concatenate([train_data['x_train'],
                                                 jnp.concatenate([actions, trajectory[:-1]], axis=-1)], axis=0)
        train_data['y_train'] = jnp.concatenate([train_data['y_train'], trajectory[1:]], axis=0)


if __name__ == '__main__':
    main()