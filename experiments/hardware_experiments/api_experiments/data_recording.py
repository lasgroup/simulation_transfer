import jax.tree_util
import numpy as np

from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.hardware.xbox_data_recording.xboxagent import CarXbox2D
from brax.training.types import Transition
import pickle

RECORDING_NAME = 'recording_sep1_1.pickle'


if __name__ == '__main__':
    controller = CarXbox2D(base_speed=1.0)
    env = CarEnv(encode_angle=False)
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
        print(action, stop, obs)
        obs = new_obs
        # if terminate:
        #    obs, _ = env.reset()
    env.close()
    observations = np.array(observations)
    next_observations = np.array(new_observations)
    actions = np.array(actions)
    transitions = Transition(
        observation=observations,
        action=actions,
        next_observation=next_observations,
        reward=np.zeros(actions.shape[0]),
        discount=np.ones(actions.shape[0])
    )

    with open(RECORDING_NAME, 'wb') as f:
        pickle.dump(transitions, f)
