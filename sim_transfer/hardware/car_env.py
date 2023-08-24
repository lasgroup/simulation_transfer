import gym
import sys
import numpy as np
from gym.spaces import Box
from typing import Optional

X_MIN_LIMIT = -2.55
X_MAX_LIMIT = 2.55
Y_MAX_LIMIT = 3.2
Y_MIN_LIMIT = -3.2


class CarEnv(gym.Env):

    def __init__(self,
                 control_frequency: float = 30,
                 max_wait_time: float = 1,
                 window_size: int = 6,
                 num_frame_stacks: int = 3,
                 port_number: int = 8,  # leftmost usb port in the display has port number 8
                 encode_angle: bool = True,
                 goal: np.ndarray = np.asarray([0.0, 0.0, 0.0])
                 ):
        super().__init__()
        sys.path.append("C:Users/Panda/Desktop/rcCarInterface/rc-car-interface/build/src/libs/pyCarController")
        import carl
        self.control_frequency = control_frequency
        self.max_wait_time = max_wait_time
        self.window_size = window_size
        self.num_frame_stacks = num_frame_stacks
        self.port_number = port_number
        self.encode_angle = encode_angle
        self.controller = carl.controller(w_size=window_size, p_number=port_number, wait_time=max_wait_time,
                                          control_freq=control_frequency)
        self.initial_reset = True
        self.controller_started = False

        self.num_frame_stacks = num_frame_stacks
        self.goal = goal
        self.max_steps = 200
        self.env_steps = 0
        high = np.ones(6 + self.encode_angle + 2 * num_frame_stacks) * np.inf
        if self.encode_angle:
            high[2:4] = 1
        high[6:] = 1

        self.observation_space = Box(low=-high, high=high, shape=(6 + self.encode_angle + 2 * num_frame_stacks,))

        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.state: np.array = np.zeros((6 + int(self.encode_angle) + 2 * num_frame_stacks,))

    def log_mocap_info(self):
        logs = self.controller.get_mocap_logs()
        logs_dictionary = {
            'last_frame': logs.last_frame,
            'total_frames': logs.total_frames,
            'recorded_frames': logs.recorded_frames,
            'requested_frames': logs.requested_frames,
            'skipped_frames': logs.skipped_frames,
            'invalid_frames': logs.invalid_frames,
        }
        import os
        from datetime import datetime
        import csv
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        file_path = os.path.join(os.getcwd(), 'logs/')
        isExist = os.path.exists(file_path)
        if not isExist:
            os.makedirs(file_path)
        file_name = os.path.join(file_path, date_time + 's.csv')
        with open(file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=logs_dictionary.keys())
            writer.writeheader()
            writer.writerow(logs_dictionary)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        # self.controller.stop()
        if not self.initial_reset:
            self.log_mocap_info()
        self.initial_reset = False
        answer = input("Press Y to continue the reset.")
        assert answer == 'Y', "environment execution aborted."
        if not self.controller_started:
            self.controller.start()
            self.controller_started = True
        current_state = self.controller.get_state()
        current_state[0:3] = current_state[0:3] - self.goal
        if self.encode_angle:
            new_state = self.get_encoded_state(current_state)
        else:
            new_state = current_state
        initial_actions = np.zeros(2 * self.num_frame_stacks)
        state = np.concatenate([new_state, initial_actions], axis=0)
        self.state = state
        return state, {}

    def close(self):
        self.controller.stop()
        self.log_mocap_info()
        self.controller_started = False

    def get_encoded_state(self, true_state):
        encoded_state = np.zeros(7)
        encoded_state[0:2] = true_state[0:2]
        encoded_state[4:] = true_state[3:]
        encoded_state[2] = np.sin(true_state[2])
        encoded_state[3] = np.cos(true_state[2])
        return encoded_state

    def step(self, action):
        assert np.shape(action) == (2,)
        self.controller.control_mode()  # sets the mode to control
        self.controller.set_command(action)  # set action
        next_state = self.controller.get_state()  # get state
        next_state[0:3] = next_state[0:3] - self.goal
        new_state = np.zeros_like(self.state)
        # if desired, encode angle
        if self.encode_angle:
            robot_state = self.get_encoded_state(next_state)
        else:
            robot_state = next_state

        dim_true_state = 6 + int(self.encode_angle)
        # set internal state for frames tacking
        new_state[0:dim_true_state] = robot_state
        num_previous_actions = 2 * (self.num_frame_stacks - 1)
        if self.num_frame_stacks > 1:
            new_state[dim_true_state:num_previous_actions + dim_true_state] = \
                self.state[2 + dim_true_state:]
        new_state[num_previous_actions + dim_true_state:] = action  # store the latest action

        reward = self.reward(self.state, action, new_state)
        terminate = self.terminate(next_state)
        self.state = new_state
        return new_state, reward, terminate, {}

    def reward(self, state, action, next_state):
        return 0.0

    def terminate(self, next_state): # TODO fix termination flag
        reached_goal = self.reached_goal(next_state, self.goal).item()
        out_of_bound = self.constraint_violation(next_state).item()
        time_out = self.env_steps >= self.max_steps

        if reached_goal:
            print("REACHED GOAL!")
        elif out_of_bound:
            print("CONSTRAINT VIOLATION!")
        elif time_out:
            print("TIMEOUT!")
        terminate = reached_goal + out_of_bound + time_out
        return terminate

    @staticmethod
    def constraint_violation(state):
        in_bounds = np.logical_and(
            np.logical_and(X_MAX_LIMIT >= state[0], state[0] >= X_MIN_LIMIT),
            np.logical_and(Y_MAX_LIMIT >= state[1], state[1] >= Y_MIN_LIMIT)
        )
        return np.where(in_bounds, 0, 1)

    @staticmethod
    def reached_goal(state, goal):
        dist = np.sqrt(np.square(state[:2] - goal[:2]).sum(-1))
        ang_dev = np.abs(state[2] - goal[2])
        speed = np.sqrt(np.square(state[..., 3:]).sum(-1))
        in_bounds = np.logical_and(np.logical_and(dist < 0.25, ang_dev < 1.0), speed < 1.5)
        return np.where(in_bounds, 1, 0)

    @staticmethod
    def normalize_theta(state):
        state[2] = ((state[2] + np.pi) % (2 * np.pi)) - np.pi
        return state
