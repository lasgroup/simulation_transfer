import jax.numpy as jnp
import jax
import time

from functools import partial
from sim_transfer.sims.dynamics_models import RaceCar, CarParams
from sim_transfer.sims.util import encode_angles, decode_angles, plot_rc_trajectory
from typing import Union, Dict, Any, Callable, Tuple, Optional, List


class RCCarSimEnv:
    max_steps: int = 200
    _dt: float = 1/30.
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, - jnp.pi / 2.])
    _init_pose: jnp.array = jnp.array([-1.04, -1.42, jnp.pi / 2.])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = 0.05 * jnp.exp(jnp.array([-3.3170326, -3.7336411, -2.7081904,
                                                -2.7841284, -2.7067015, -1.4446207]))

    def __init__(self, ctrl_cost_weight: float = 0.005, encode_angle: bool = False, use_obs_noise: bool = True,
                 seed: int = 230492394):
        self.dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)

        # initialize dynamics and observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=self.encode_angle)
        self._dynamics_params = CarParams(use_blend=0.0)  # TODO allow setting the params
        self._next_step_fn = jax.jit(partial(self._dynamics_model.next_step, params=self._dynamics_params))

        self.use_obs_noise = use_obs_noise

        # initialize reward model
        self._reward_model = None  # TODO

        # initialize time and state
        self._time: int = 0
        self._state: jnp.array = jnp.zeros(self.dim_state)

    def reset(self, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.array:
        """ Resets the environment to a random initial state close to the initial pose """
        rng_key = self.rds_key if rng_key is None else rng_key

        # sample random initial state
        key_pos, key_theta, key_vel = jax.random.split(rng_key, 3)
        init_pos = self._init_pose[:2] + jax.random.uniform(key_pos, shape=(2,), minval=-0.10, maxval=0.10)
        init_theta = self._init_pose[2:] + \
                     jax.random.uniform(key_pos, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi)
        init_vel = jnp.zeros((3,)) + jnp.array([0.005, 0.005, 0.02]) * jax.random.normal(key_vel, shape=(3,))
        init_state = jnp.concatenate([init_pos, init_theta, init_vel])

        self._state = init_state
        self._time = 0
        return self._state

    def step(self, action: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None) \
            -> Tuple[jnp.array, float, bool, Dict[str, Any]]:
        """ Performs one step in the environment """

        assert action.shape[-1:] == self.dim_action
        assert jnp.all(-1 <= action) and jnp.all(action <= 1), "action must be in [-1, 1]"
        rng_key = self.rds_key if rng_key is None else rng_key

        # compute next state
        self._state = self._next_step_fn(self._state, action)
        self._time += 1
        obs = self._state_to_obs(self._state, rng_key=rng_key)

        # compute reward
        reward = 10.   # TODO

        # check if done
        done = self._time >= self.max_steps

        # return observation, reward, done, info
        return obs, reward, done, {'time': self._time, 'state': self._state}

    def _state_to_obs(self, state: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.array:
        """ Adds observation noise to the state """
        assert state.shape[-1] == 6
        rng_key = self.rds_key if rng_key is None else rng_key

        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jax.random.normal(rng_key, shape=self._state.shape)
        else:
            obs = state

        # encode angle to sin(theta) and cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(state, self._angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (obs.shape[-1] == 6 and not self.encode_angle)
        return obs


    @property
    def rds_key(self) -> jax.random.PRNGKey:
        self._rds_key, key = jax.random.split(self._rds_key)
        return key

    @property
    def time(self) -> float:
        return self._time


if __name__ == '__main__':
    env = RCCarSimEnv(encode_angle=False,
                      use_obs_noise=False,)

    t_start = time.time()

    s = env.reset()
    traj = [s]
    for i in range(200):
        t = i / 30.
        a = jnp.array([- 0.8 * jnp.cos(2 * t), 0.5 / (t+1)])
        s, _, _, _ = env.step(a)
        traj.append(s)

    duration = time.time() - t_start
    print(f'Duration of trajectory sim {duration} sec')
    traj = jnp.stack(traj)

    plot_rc_trajectory(traj)
