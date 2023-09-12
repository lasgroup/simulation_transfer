import pickle
import time

import jax.numpy as jnp
import jax.tree_util as jtu
from brax.training.types import Transition

from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.util import plot_rc_trajectory

if __name__ == '__main__':
    ENCODE_ANGLE = True
    SAVE_TRANSITIONS = True
    NUM_FRAMES_STACKED = 3
    state_dim = 7 if ENCODE_ANGLE else 6
    action_dim = 2
    DISCOUNT = jnp.array(0.99)
    env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                      action_delay=0.07,
                      use_tire_model=True,
                      use_obs_noise=True)

    t_start = time.time()
    obs = env.reset()
    traj = [obs]
    rewards = []
    actions = []
    transitions = []
    last_actions = jnp.zeros(action_dim * NUM_FRAMES_STACKED)
    for i in range(120):
        t = i / 30.
        a = jnp.array([- 1 * jnp.cos(1.0 * t), 0.8 / (t + 1)])
        new_obs, reward, done, info = env.step(a)
        traj.append(new_obs)
        actions.append(a)
        rewards.append(reward)
        stacked_observation = jnp.concatenate([obs, last_actions])

        # Here we roll last actions and append new action
        last_actions = jnp.roll(last_actions, shift=action_dim)
        last_actions = last_actions.at[0:action_dim].set(a)
        stacked_next_observation = jnp.concatenate([new_obs, last_actions])
        transitions.append(Transition(observation=stacked_observation,
                                      action=a,
                                      reward=jnp.array(reward),
                                      discount=DISCOUNT,
                                      next_observation=stacked_next_observation,
                                      ))

        obs = new_obs

    # Stack and save transitions with pickle
    transitions = jtu.tree_map(lambda *xs: jnp.stack(xs), *transitions)
    if SAVE_TRANSITIONS:
        with open('../../../experiments/rl_on_hardware/transitions.pkl', 'wb') as f:
            pickle.dump(transitions, f)

    duration = time.time() - t_start
    print(f'Duration of trajectory sim {duration} sec')
    traj = jnp.stack(traj)
    actions = jnp.stack(actions)

    plot_rc_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)
