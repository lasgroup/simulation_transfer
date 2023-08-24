from typing import Optional, Tuple
import numpy as np
from joy_stick_api.xbox_joystick_factory import XboxJoystickFactory


class CarXbox2D(object):
    """SpotXbox2D class provides mapping between xbox controller commands and Spot2D actions.
    """

    def __init__(self, base_speed: float = 1., base_angular: float = 1.):
        super().__init__()
        self.joy = XboxJoystickFactory.get_joystick()
        self.base_speed = base_speed
        self.base_angular = base_angular
    
    def _move(self, right_x, left_y):
        """Commands the robot with a velocity command based on left/right stick values.

        Args:
            left_x: X value of left stick.
            left_y: Y value of left stick.
        """

        # Stick left_x controls robot v_y
        steering = -right_x * self.base_speed

        # Stick left_y controls robot v_x
        throttle = left_y * self.base_speed

        return np.array([steering, throttle])

    def act(self, obs: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
        """Controls robot from an Xbox controller.

        Mapping
        Button Combination    -> Functionality
        --------------------------------------
        LB + RB + B           -> Return None
          Left Stick          -> Move
          Right Stick         -> Turn

        """

        right_x = self.joy.right_x()
        left_y = self.joy.left_y()


        if right_x != 0.0 or left_y != 0.0:
            return self._move(right_x, left_y), self.joy.B()
        else:
            return self._move(0.0, 0.0), self.joy.B()


if __name__ == '__main__':
    from sim_transfer.hardware.car_env import CarEnv
    controller = CarXbox2D(base_speed=0.5)
    env = CarEnv()
    obs, _ = env.reset()
    stop = False
    observations = []
    new_observations = []
    actions = []
    while not stop:
        action, stop = controller.act(obs)
        new_obs, reward, terminate, info = env.step(action)
        observations.append(obs)
        new_observations.append(new_obs)
        actions.append(action)
        print(action, stop)
        obs = new_obs
        # if terminate:
        #    obs, _ = env.reset()
    observations = np.array(observations)
    new_observations = np.array(new_observations)
    actions = np.array(actions)
    env.close()



