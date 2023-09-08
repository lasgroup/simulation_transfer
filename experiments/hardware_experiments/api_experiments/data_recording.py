import jax.tree_util
import numpy as np

from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.hardware.xbox_data_recording.xboxagent import CarXbox2D
from brax.training.types import Transition
import pickle
import time
from matplotlib import pyplot as plt

RECORDING_NAME = 'recording_sep8_carcheck_debug.pickle'
AR1_param = 0.0
ACT_NOISE_BOUND = 0.0

if __name__ == '__main__':
    controller = CarXbox2D(base_speed=1.0)
    env = CarEnv(encode_angle=False, max_throttle=0.5, control_time_ms=32.)
    obs, _ = env.reset()
    env.step(np.zeros(2))

    stop = False
    t_prev = time.time()
    observations = []
    new_observations = []
    actions = []
    rewards = []
    time_diffs = []
    itr = 1
    prev_action_noise = np.zeros(2)

    while not stop:
        # get action from controller and add a small amount of noise to them
        action, stop = controller.act(obs)

        # add AR(1) action noise
        action_noise = AR1_param * prev_action_noise + (1 - AR1_param) * \
                       np.random.uniform(-ACT_NOISE_BOUND, ACT_NOISE_BOUND, size=(2,))
        prev_action_noise = action_noise
        action = np.clip(action + action_noise, -1, 1)

        # perform step on env
        new_obs, reward, terminate, info = env.step(action)

        t = time.time()
        time_diff = t - t_prev
        t_prev = t

        print('itr:', itr, 'act:', action, 'obs:', obs, 't:', f'{time_diff} sec')

        # append transitions, actions and rewards to trajectory
        observations.append(obs)
        new_observations.append(new_obs)
        actions.append(action)
        rewards.append(reward)
        time_diffs.append(time_diff)

        obs = new_obs
        itr += 1

    env.close()
    observations = np.array(observations)
    next_observations = np.array(new_observations)
    actions = np.array(actions)
    time_diffs = np.array(time_diffs)
    transitions = Transition(
        observation=observations,
        action=actions,
        next_observation=next_observations,
        reward=np.zeros(actions.shape[0]),
        discount=np.ones(actions.shape[0]),
        extras={'time_diff': time_diffs}
    )

    print('Avg time per iter:', np.mean(time_diffs))
    plt.plot(np.arange(time_diffs.shape[0]), time_diffs)
    plt.title('time diffs')
    plt.show()

    with open(RECORDING_NAME, 'wb') as f:
        pickle.dump(transitions, f)
    print(f'Saved recordings to: {RECORDING_NAME}')

    with open(RECORDING_NAME, 'rb') as f:
        transitions = pickle.load(f)


    from sim_transfer.sims.util import plot_rc_trajectory
    print(RECORDING_NAME)
    print(transitions.observation.shape)

    traj = transitions.observation[:400, :6]
    acts = transitions.action[:400]
    plot_rc_trajectory(traj, acts)