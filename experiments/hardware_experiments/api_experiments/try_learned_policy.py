import pickle
import time

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import wandb
from matplotlib import pyplot as plt

from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.rl.rl_on_offline_data import RLFromOfflineData
from sim_transfer.sims.util import decode_angles
from sim_transfer.sims.util import plot_rc_trajectory


def run_with_learned_policy(filename_policy: str,
                            filename__bnn_model: str,
                            closed_loop: bool = False):
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
    wandb.init(
        project="Race car test MBRL",
        group='hardware test',
        entity='trevenl'
    )

    with open(filename__bnn_model, 'rb') as handle:
        bnn_model = pickle.load(handle)
        """
        We predict state difference:
        
        delta_x_dist = bnn_model.predict_dist(z, include_noise=self.include_noise)
        delta_x = delta_x_dist.sample(seed=key_sample_x_next)
        
        z is of the shape (-1, state_dim + num_frame_stack * action_dim + action_dim)
        """

    policy = rl_from_offline_data.prepare_policy(params=None, filename=filename_policy)

    # replay action sequence on car
    env = CarEnv(encode_angle=True, num_frame_stacks=0)
    obs, _ = env.reset()
    print(obs)
    initial_obs = obs
    stop = False
    observations = []
    env.step(np.zeros(2))
    t_prev = time.time()
    actions = []

    num_frame_stack = 3
    action_dim = 2
    state_dim = 7

    stacked_actions = jnp.zeros(shape=(num_frame_stack * action_dim,))
    time_diffs = []

    """
    Simulate the car on the learned policy
    """
    sim_obs = obs
    sim_stacked_actions = jnp.zeros(shape=(num_frame_stack * action_dim,))
    sim_key = jr.PRNGKey(0)

    all_sim_obs = []
    all_sim_actions = []

    for i in range(200):
        sim_action = policy(jnp.concatenate([sim_obs, sim_stacked_actions], axis=-1))
        sim_action = np.array(sim_action)

        z = jnp.concatenate([sim_obs, sim_stacked_actions, sim_action], axis=-1)
        z = z.reshape(1, -1)
        delta_x_dist = bnn_model.predict_dist(z, include_noise=True)
        sim_key, subkey = jr.split(sim_key)
        delta_x = delta_x_dist.sample(seed=subkey)
        sim_obs = sim_obs + delta_x.reshape(-1)

        # Now we shift the actions
        sim_stacked_actions = jnp.roll(sim_stacked_actions, shift=action_dim)
        sim_stacked_actions = sim_stacked_actions.at[:action_dim].set(sim_action)
        all_sim_actions.append(sim_action)
        all_sim_obs.append(sim_obs)

    sim_observations_for_plotting = np.stack(all_sim_obs, axis=0)
    sim_actions_for_plotting = np.stack(all_sim_actions, axis=0)
    fig, axes = plot_rc_trajectory(sim_observations_for_plotting,
                                   sim_actions_for_plotting,
                                   encode_angle=True,
                                   show=True)
    wandb.log({'Trajectory_on_learned_model': wandb.Image(fig)})

    if closed_loop:
        for i in range(200):
            action = policy(jnp.concatenate([obs, stacked_actions], axis=-1))
            action = np.array(action)
            actions.append(action)
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
    if not closed_loop:
        for i in range(200):
            action = all_sim_actions[i]
            action = np.array(action)
            actions.append(action)
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
            if terminate:
                break

    env.close()
    observations = np.array(observations)
    time_diffs = np.array(time_diffs)
    plt.plot(time_diffs)
    plt.title('time diffs')
    plt.show()

    # We plot the true trajectory
    observations_for_plotting = np.stack(observations, axis=0)
    actions_for_plotting = np.stack(actions, axis=0)
    fig, axes = plot_rc_trajectory(observations_for_plotting,
                                   actions_for_plotting,
                                   encode_angle=True,
                                   show=True)
    wandb.log({'Trajectory_on_true_model': wandb.Image(fig)})

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
    file_policy = 'parameters.pkl'
    file_bnn_model = 'bnn_model.pkl'
    run_with_learned_policy(filename_policy=file_policy,
                            filename__bnn_model=file_bnn_model,
                            closed_loop=True)
