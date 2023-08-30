import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.sims.util import plot_rc_trajectory


ENCODE_ANGLE = False
env = CarEnv(encode_angle=ENCODE_ANGLE)


def simulate_system_response(steps=120, velocity_max=0.8, num_runs=1):
    all_time_values = []
    all_response_values = []
    all_actions = []
    for run in range(num_runs):
        time_values = []
        response_values = []
        actions = []
        time.sleep(5)

        obs, _ = env.reset()
        start_time = time.time()
        elapsed_time = 0
        for step in range(steps):
            current_time = time.time() - start_time
            action = np.array([-1 * np.cos(step/30.0), velocity_max / (step/30.0 + 1)])
            # action = np.array([0, 0.8])
            t = time.time()
            next_obs, reward, done, info = env.step(action)
            print('time to set command', time.time() - t)
            t = time.time()
            response_values.append(obs[: 6 + int(ENCODE_ANGLE)])
            print('time to get state', time.time() - t)
            obs = next_obs
            time_values.append(current_time)
            elapsed_time = current_time
            print(current_time)
            print(action)
            actions.append(action)

        all_time_values.append(time_values)
        all_response_values.append(response_values)
        all_actions.append(actions)
        env.close()
    return all_time_values, all_response_values, all_actions


num_runs = 1
time_values, response_values, all_actions = simulate_system_response(steps=50, num_runs=num_runs, velocity_max=0.5)
#
response_array = np.stack([np.array(response_value).reshape(-1, 6 + int(ENCODE_ANGLE)) for response_value in
                   response_values])

actions = np.stack([np.array(response_value).reshape(-1, env.action_space.shape[0]) for response_value in
                   all_actions])

for i in range(num_runs):
    plot_rc_trajectory(response_array[i], actions[i], encode_angle=ENCODE_ANGLE)
# time_values = [np.array(time_value).reshape(-1, 1) for time_value in time_values]
# num_dims = 6
# for dim in range(num_dims):
#     plt.figure(figsize=(10, 8))
#     for run in range(num_runs):
#         plt.plot(time_values[run] * 1000, response_array[run][:, dim], label=f'Run {run + 1}')
#         plt.axhline(response_array[run][0, dim], color='red', linestyle='dashed', label='initial state')
#     plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=20))
#     plt.xlabel("Time (milli seconds)")
#     plt.ylabel(f"State {dim + 1}")
#     plt.title(f'Step response - Dimension {dim + 1}')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'dimension_{dim + 1}_response.pdf')
#     plt.close()
