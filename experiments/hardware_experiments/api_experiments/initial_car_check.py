import pickle
import numpy as np
import time
from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.sims.util import decode_angles
from matplotlib import pyplot as plt

""" 
Script for checking that the rccar is set up properly.
Instructions:
1) Put car on start location
2) Execute script
3) Compare plots whether trajecory roughly matches the recorded trajectory
"""


def main():
    # load recorded trajectory
    RECORDING_NAME = f'recording_sep4_1.pickle'
    with open(RECORDING_NAME, 'rb') as f:
        rec_traj = pickle.load(f)
    rec_observations = rec_traj.observation[:200]
    rec_actions = rec_traj.action[:200]

    # replay action sequence on car
    env = CarEnv(encode_angle=True)
    obs, _ = env.reset()
    env.step(np.zeros(2))
    t_prev = time.time()
    observations = [obs]
    rewards = []
    time_diffs = []
    for i in range(rec_actions.shape[0]):
        action = rec_actions[i]
        obs, reward, terminate, info = env.step(action)
        t = time.time()
        time_diff = t - t_prev
        t_prev = t
        print(i, action, reward, time_diff)
        time_diffs.append(time_diff)
        rewards.append(reward)
        observations.append(obs)
    env.close()
    observations = np.array(observations)
    rewards = np.array(rewards)
    time_diffs = np.array(time_diffs)

    plt.plot(time_diffs)
    plt.title('time diffs')
    plt.show()

    plt.plot(rewards)
    plt.title('reward')
    plt.show()

    observations = decode_angles(observations, angle_idx=2)
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
    print('finished plotting')
    fig.show()

if __name__ == '__main__':
    main()