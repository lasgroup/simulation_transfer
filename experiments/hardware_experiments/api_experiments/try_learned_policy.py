import pickle

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import time

from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.rl.rl_on_offline_data import RLFromOfflineData
from sim_transfer.sims.util import decode_angles



def run_with_learned_policy(filename: str):
    """
    Num stacked frames: 3
    """
    car_reward_kwargs = dict(encode_angle=True,
                             ctrl_cost_weight=0.005,
                             margin_factor=20)

    NUM_ENV_STEPS_BETWEEN_UPDATES = 16
    NUM_ENVS = 64
    sac_num_env_steps = 1_000_000
    horizon_len = 50

    SAC_KWARGS = dict(num_timesteps=sac_num_env_steps,
                      num_evals=20,
                      reward_scaling=10,
                      episode_length=horizon_len,
                      episode_length_eval=2 * horizon_len,
                      action_repeat=1,
                      discounting=0.99,
                      lr_policy=3e-4,
                      lr_alpha=3e-4,
                      lr_q=3e-4,
                      num_envs=NUM_ENVS,
                      batch_size=64,
                      grad_updates_per_step=NUM_ENV_STEPS_BETWEEN_UPDATES * NUM_ENVS,
                      num_env_steps_between_updates=NUM_ENV_STEPS_BETWEEN_UPDATES,
                      tau=0.005,
                      wd_policy=0,
                      wd_q=0,
                      wd_alpha=0,
                      num_eval_envs=2 * NUM_ENVS,
                      max_replay_size=5 * 10 ** 4,
                      min_replay_size=2 ** 11,
                      policy_hidden_layer_sizes=(64, 64),
                      critic_hidden_layer_sizes=(64, 64),
                      normalize_observations=True,
                      deterministic_eval=True,
                      wandb_logging=True)

    rl_from_offline_data = RLFromOfflineData(
        sac_kwargs=SAC_KWARGS,
        car_reward_kwargs=car_reward_kwargs)

    policy = rl_from_offline_data.prepare_policy(params=None, filename=filename)

    # replay action sequence on car
    env = CarEnv(encode_angle=True, num_frame_stacks=0)
    obs, _ = env.reset()
    initial_obs = obs
    stop = False
    observations = [obs]
    env.step(np.zeros(2))
    t_prev = time.time()
    actions = []

    num_frame_stack = 3
    action_dim = 2
    stacked_actions = jnp.zeros(shape=(num_frame_stack * action_dim,))
    time_diffs = []


    for i in range(200):
        action = policy(jnp.concatenate([obs, stacked_actions], axis=-1))
        action = np.array(action)
        obs, reward, terminate, info = env.step(action)
        t = time.time()
        time_diff = t - t_prev
        t_prev = t
        print(i, action, reward, time_diff)
        time_diffs.append(time_diff)
        observations.append(obs)

        # Now we shift the actions
        stacked_actions = jnp.roll(stacked_actions, shift=action_dim)
        stacked_actions = stacked_actions.at[:action_dim].set(action)

    env.close()
    observations = np.array(observations)
    time_diffs = np.array(time_diffs)
    plt.plot(time_diffs)
    plt.title('time diffs')
    plt.show()

    # Now we run the simulation on the learned model

    observations = decode_angles(observations, angle_idx=2)
    # comparison plot recorded and new traj
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

    axes[0].plot(observations[:, 0], color='orange', label='new')
    axes[1].plot(observations[:, 1], color='orange')
    axes[2].plot(observations[:, 2], color='orange')

    axes[0].set_title("x pos")
    axes[1].set_title("y pos")
    axes[2].set_title("theta")
    fig.legend()
    print('finished plotting')
    fig.show()



if __name__ == '__main__':
    filename = 'parameters.pkl'
    run_with_learned_policy(filename=filename)
