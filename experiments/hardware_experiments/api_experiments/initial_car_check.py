import pickle
import numpy as np
import time
from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.util import decode_angles
from matplotlib import pyplot as plt
from sim_transfer.sims.util import plot_rc_trajectory, rotate_coordinates


""" 
Script for checking that the rccar is set up properly.
Instructions:
1) Put car on start location
2) Execute script
3) Compare plots whether trajecory roughly matches the recorded trajectory
"""


ENCODE_ANGLE = True
SIMULATION = False

def main():
    # load recorded trajectory
    RECORDING_NAME = f'recording_check_car2_sep29.pickle'
    with open(RECORDING_NAME, 'rb') as f:
        rec_traj = pickle.load(f)
    rec_observations = rec_traj.observation
    rec_actions = rec_traj.action

    # replay action sequence on car
    if SIMULATION:
        env = RCCarSimEnv(encode_angle=ENCODE_ANGLE, use_tire_model=True)
        obs = env.reset()
    else:
        env = CarEnv(encode_angle=ENCODE_ANGLE, max_throttle=0.4, control_time_ms=30.68,
                     num_frame_stacks=0, car_id=2)
        obs, _ = env.reset()
    env.step(np.zeros(2))
    t_prev = time.time()
    observations = []
    actions = []
    rewards = []
    time_diffs = []
    itr = 1
    for i in range(rec_actions.shape[0]):
        action = rec_actions[i]

        # perform step on env
        obs, reward, terminate, info = env.step(action)

        t = time.time()
        time_diff = t - t_prev
        t_prev = t

        print('itr:', itr, 'act:', action, 'obs:', obs, 't:', f'{time_diff} sec')

        # append transitions, actions and rewards to trajectory
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        time_diffs.append(time_diff)

        itr += 1
    if not SIMULATION:
        env.close()
    observations = np.array(observations)
    actions = np.array(actions)
    time_diffs = np.array(time_diffs)

    print('Avg time per iter:', np.mean(time_diffs))
    plt.plot(time_diffs)
    plt.title('time diffs')
    plt.show()

    plt.plot(rewards)
    plt.title('reward')
    plt.show()

    plt.plot(rec_actions[:, 1], label='throttle')
    plt.plot(rec_actions[:, 0], label='steering')
    plt.legend()
    plt.title('actions')
    plt.show()


    plt.title('reward')
    plt.show()

    plot_rc_trajectory(observations[..., :(7 if ENCODE_ANGLE else 6)], actions,
                       encode_angle=ENCODE_ANGLE,
                       show=True)

    if ENCODE_ANGLE:
        observations = decode_angles(observations, angle_idx=2)

    rec_observations = rotate_coordinates(rec_observations, encode_angle=True)
    observations = rotate_coordinates(observations, encode_angle=True)
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