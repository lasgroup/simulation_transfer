import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import sys

from sim_transfer.hardware.car_env import CarEnv


def simulate_system_response(duration=0.5, step_value=1.0, num_runs=3):
    all_time_values = []
    all_response_values = []
    all_raw_response_values = []
    env = CarEnv()
    for run in range(num_runs):
        answer = input("Press Y to continue run {}...".format(run + 1))
        if answer != 'Y':
            continue
        time_values = []
        response_values = []
        raw_response_values = []

        time.sleep(5)

        obs, _ = env.reset()

        time.sleep(3)

        start_time = time.time()
        elapsed_time = 0
        action = np.zeros(2)
        while elapsed_time < duration:
            current_time = time.time() - start_time
            command = step_value if current_time < duration else 0.0
            action[-1] = command
            t = time.time()
            next_state, reward, terminate, info = env.step(action)
            print('time to set command', time.time() - t)
            print('time to get state', time.time() - t)
            raw_response_values.append(env.controller.get_raw_state())
            time_values.append(current_time)
            elapsed_time = current_time
            print(current_time)

        all_time_values.append(time_values)
        all_response_values.append(response_values)
        all_raw_response_values.append(raw_response_values)
        env.close()
    return all_time_values, all_response_values, all_raw_response_values


num_runs = 3
time_values, response_values, raw_response_values = simulate_system_response(num_runs=num_runs)

response_array = [np.array(response_value) for response_value in response_values]
raw_response_array = [np.array(response) for response in raw_response_values]
time_values = [np.array(time_value) for time_value in time_values]
num_dims = 6
for dim in range(num_dims):
    plt.figure(figsize=(10, 8))
    for run in range(num_runs):
        plt.plot(time_values[run] * 1000, response_array[run][:, dim], label=f'Run {run + 1}')
        plt.axhline(response_array[run][0, dim], color='red', linestyle='dashed', label='initial state')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=20))
    plt.xlabel("Time (milli seconds)")
    plt.ylabel(f"State {dim + 1}")
    plt.title(f'Step response - Dimension {dim + 1}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'dimension_{dim + 1}_response.pdf')
    plt.close()

for dim in range(num_dims):
    plt.figure(figsize=(10, 8))
    for run in range(num_runs):
        plt.plot(time_values[run] * 1000, raw_response_array[run][:, dim], label=f'Run {run + 1}')
        plt.axhline(response_array[run][0, dim], color='red', linestyle='dashed', label='initial state')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=20))
    plt.xlabel("Time (milli seconds)")
    plt.ylabel(f"State {dim + 1}")
    plt.title(f'Step response - Dimension {dim + 1}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'dimension_{dim + 1}_raw_response.pdf')
    plt.close()
