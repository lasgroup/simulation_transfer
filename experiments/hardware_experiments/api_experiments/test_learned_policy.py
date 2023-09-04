import pickle

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.rl.rl_on_offline_data import RLFromOfflineData


def run_with_learned_policy(filename: str,
                            recording_name: str, ):
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

    # load recorded trajectory
    RECORDING_NAME = f'{recording_name}.pickle'
    with open(RECORDING_NAME, 'rb') as f:
        rec_traj = pickle.load(f)
    rec_observations = rec_traj.observation[:200]
    rec_actions = rec_traj.action[:200]

    # replay action sequence on car
    env = CarEnv(encode_angle=True)
    obs, _ = env.reset()
    stop = False
    observations = [obs]
    actions = []

    num_frame_stack = 3
    action_dim = 2
    stacked_actions = jnp.zeros(shape=(num_frame_stack * action_dim,))

    for i in range(rec_actions.shape[0]):
        action = policy(jnp.concatenate([obs, stacked_actions], axis=-1))
        print(action)
        obs, reward, terminate, info = env.step(action)
        observations.append(obs)

        # Now we shift the actions
        stacked_actions = jnp.roll(stacked_actions, shift=action_dim)
        stacked_actions = stacked_actions.at[:action_dim].set(action)

    env.close()
    observations = np.array(observations)

    # comparison plot recorded and new traj
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    axes[0].plot(rec_observations[:, 0], color='blue', label='rec')
    axes[1].plot(rec_observations[:, 1], color='blue')
    axes[2].plot(rec_observations[:, 2], color='blue')

    axes[0].plot(observations[:, 0], color='orange', label='new')
    axes[1].plot(observations[:, 1], color='orange')
    axes[2].plot(observations[:, 2], color='orange')

    axes[0].set_title("x pos")
    axes[1].set_title("y pos")
    axes[2].set_title("theta")
    fig.legend()

    fig.show()


if __name__ == '__main__':
    filename = 'learned_policy_params.pickle'
    recording_name = 'offline_policy'
    run_with_learned_policy(filename=filename,
                            recording_name=recording_name)
