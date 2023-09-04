import pickle
import numpy as np
from sim_transfer.hardware.car_env import CarEnv
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
    env = CarEnv(encode_angle=False)
    obs, _ = env.reset()
    stop = False
    observations = [obs]
    actions = []
    for i in range(rec_actions.shape[0]):
        action = rec_actions[i]
        print(action)
        obs, reward, terminate, info = env.step(action)
        observations.append(obs)
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
    main()