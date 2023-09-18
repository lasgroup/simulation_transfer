import os
import time
from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.hardware_experiments.api_experiments._default_params import SAC_KWARGS
from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.rl.rl_on_offline_data import RLFromOfflineData
from sim_transfer.sims.util import plot_rc_trajectory

ENTITY = 'trevenl'


class RunSpec(NamedTuple):
    group_name: str
    run_id: str


def run_all_hardware_experiments(project_name_load: str,
                                 project_name_save: str | None = None, ):
    api = wandb.Api()
    project_name = ENTITY + '/' + project_name_load
    local_dir = "saved_data"
    dir_to_save = 'models'

    project_name_save = project_name_load if project_name_save is None else project_name_save

    # check if directory exists
    if not os.path.exists(local_dir):
        # if directory does not exist, create it
        os.makedirs(local_dir)

    runs_spec = []

    # Download all models
    runs = api.runs(project_name)
    for run in runs:
        for file in run.files():
            if file.name.startswith(dir_to_save):
                file.download(replace=True, root=os.path.join(local_dir, run.group, run.id))
                runs_spec.append(RunSpec(group_name=run.group,
                                         run_id=run.id))

    # Run all models on hardware
    for run_spec in runs_spec:
        # We open the file with pickle
        pre_path = os.path.join(local_dir, run_spec.group_name, run_spec.run_id)
        policy_name = 'parameters.pkl'
        bnn_name = 'bnn_model.pkl'

        with open(os.path.join(pre_path, bnn_name), 'rb') as handle:
            bnn_model = pickle.load(handle)

        with open(os.path.join(pre_path, policy_name), 'rb') as handle:
            policy_params = pickle.load(handle)

        run_with_learned_policy(policy_params=policy_params,
                                bnn_model=bnn_model,
                                project_name=project_name_save,
                                group_name=run_spec.group_name,
                                run_id=run_spec.run_id)


def run_with_learned_policy(policy_params,
                            bnn_model,
                            project_name: str,
                            group_name: str,
                            run_id: str,
                            ):
    """
    Num stacked frames: 3
    """
    car_reward_kwargs = dict(encode_angle=True,
                             ctrl_cost_weight=0.005,
                             margin_factor=20)

    rl_from_offline_data = RLFromOfflineData(
        sac_kwargs=SAC_KWARGS,
        car_reward_kwargs=car_reward_kwargs)
    wandb.init(
        project=project_name,
        group=group_name,
        entity=ENTITY,
        id=run_id,
        resume="allow",
    )
    policy = rl_from_offline_data.prepare_policy(params=policy_params)

    # replay action sequence on car
    env = CarEnv(encode_angle=True, num_frame_stacks=0, max_throttle=0.5,
                 control_time_ms=27.9)
    obs, _ = env.reset()
    print(obs)
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

    all_sim_stacked_actions = []
    all_stacked_actions = []

    rewards = []

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

        # Now we shift the actions
        stacked_actions = jnp.roll(stacked_actions, shift=action_dim)
        stacked_actions = stacked_actions.at[:action_dim].set(action)

        observations.append(obs)
        rewards.append(reward)
        all_stacked_actions.append(stacked_actions)

        if terminate:
            break

    print('We end with simulation')
    env.close()
    observations = np.array(observations)
    actions = np.array(actions)
    time_diffs = np.array(time_diffs)

    print('Avg time per iter:', np.mean(time_diffs[1:]))
    plt.plot(time_diffs[1:])
    plt.title('time diffs')
    plt.show()

    # Here we compute the reward of the policy
    print('We calculating rewards')
    rewards = np.array(rewards)
    reward_from_observations = np.sum(rewards)
    reward_terminal = info['terminal_reward']
    total_reward = reward_from_observations + reward_terminal
    print('Terminal reward: ', reward_terminal)
    print('Reward from observations: ', reward_from_observations)
    print('Total reward: ', total_reward)

    wandb.log({
        "terminal_reward": reward_terminal,
        "reward_from_observations": reward_from_observations,
        "total_reward": total_reward
    })

    # We plot the error between the predicted next state and the true next state on the true model
    all_stacked_actions = np.stack(all_stacked_actions, axis=0)
    extended_state = np.concatenate([observations, all_stacked_actions], axis=-1)
    state_action_pairs = np.concatenate([extended_state, actions], axis=-1)

    all_inputs = state_action_pairs[:-1, :]
    target_outputs = observations[1:, :] - observations[:-1, :]

    """
    We test the model error on the predicted trajectory
    """
    print('We test the model error on the predicted trajectory')
    all_outputs = bnn_model.predict_dist(all_inputs, include_noise=False)
    sim_key, subkey = jr.split(sim_key)
    delta_x = all_outputs.sample(seed=subkey)

    assert delta_x.shape == target_outputs.shape
    # Random data for demonstration purposes
    data = delta_x - target_outputs
    fig, axes = plot_error_on_the_trajectory(data)
    wandb.log({'Error of state difference prediction': wandb.Image(fig)})

    # We plot the true trajectory
    fig, axes = plot_rc_trajectory(observations,
                                   actions,
                                   encode_angle=True,
                                   show=True)
    wandb.log({'Trajectory_on_true_model': wandb.Image(fig)})

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
        all_sim_stacked_actions.append(sim_stacked_actions)

    sim_observations_for_plotting = np.stack(all_sim_obs, axis=0)
    sim_actions_for_plotting = np.stack(all_sim_actions, axis=0)
    fig, axes = plot_rc_trajectory(sim_observations_for_plotting,
                                   sim_actions_for_plotting,
                                   encode_angle=True,
                                   show=True)
    wandb.log({'Trajectory_on_learned_model': wandb.Image(fig)})
    wandb.finish()
    return observations, actions


def plot_error_on_the_trajectory(data):
    # Create a figure with 8 subplots arranged in 2x4
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))

    # Flatten the axes for easy iteration
    axes = axes.flatten()

    # Plot each dimension on a separate subplot
    for i in range(7):
        axes[i].plot(data[:, i], label=f"Dimension {i + 1}")
        axes[i].legend()
        axes[i].set_title(f"Error evolution of Dimension {i + 1}")

    # Plot the evolution of all states on the last subplot
    for i in range(7):
        axes[7].plot(data[:, i], label=f"Dimension {i + 1}")
    axes[7].legend()
    axes[7].set_title("Evolution of All States")

    # Adjust layout for better view
    plt.tight_layout()
    return fig, axes


if __name__ == '__main__':
    import pickle

    filename_policy = 'parameters.pkl'
    filename_bnn_model = 'bnn_model.pkl'

    with open(filename_bnn_model, 'rb') as handle:
        bnn_model = pickle.load(handle)

    with open(filename_policy, 'rb') as handle:
        policy_params = pickle.load(handle)

    observations_for_plotting, actions_for_plotting = run_with_learned_policy(bnn_model=bnn_model,
                                                                              policy_params=policy_params,
                                                                              project_name='Test',
                                                                              group_name='MyGroup',
                                                                              run_name='Butterfly'
                                                                              )
