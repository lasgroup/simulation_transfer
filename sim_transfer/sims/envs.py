import jax.numpy as jnp
import jax
import time

from functools import partial
from sim_transfer.sims.dynamics_models import RaceCar, CarParams
from sim_transfer.sims.util import encode_angles, decode_angles, plot_rc_trajectory
from typing import Union, Dict, Any, Callable, Tuple, Optional, List


class ToleranceReward:
    def __init__(self, lower_bound: float, upper_bound: float, margin_coef: float, value_at_margin: float):
        self.bounds = [lower_bound, upper_bound]
        self.margin = margin_coef * (upper_bound - lower_bound)
        self.value_at_margin = value_at_margin

        if lower_bound > upper_bound:
            raise ValueError('Lower bound must be <= upper bound.')
        if margin_coef < 0:
            raise ValueError('`margin` must be non-negative.')

    def forward(self, x: jnp.array) -> jnp.array:
        lower, upper = self.bounds
        in_bounds = (x >= lower) & (x <= upper)
        if self.margin == 0:
            return jnp.where(in_bounds, 1.0, 0.0)
        else:
            d = jnp.where(x < lower, lower - x, x - upper) / self.margin
            return jnp.where(in_bounds, 1.0, self.value_at_margin * self._smooth_ramp(d))

    def _smooth_ramp(self, x: jnp.array) -> jnp.array:
        scale = jnp.sqrt(1 / self.value_at_margin - 1)
        return 1 / ((x * scale) ** 2 + 1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class RCCarEnvReward:
    _angle_idx: int = 2
    dim_action: Tuple[int] = (2,)

    def __init__(self, goal: jnp.array, encode_angle: bool = False, ctrl_cost_weight: float = 0.005):
        self.goal = goal
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle = encode_angle

        self.tolerance_pos = ToleranceReward(lower_bound=0.0, upper_bound=0.1, margin_coef=5,
                                             value_at_margin=0.2)
        self.tolerance_theta = ToleranceReward(lower_bound=0.0, upper_bound=0.1, margin_coef=5,
                                               value_at_margin=0.2)


    def forward(self, obs: jnp.array, action: jnp.array, next_obs: jnp.array):
        """ Computes the reward for the given transition """
        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(obs, next_obs)
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl
        return reward

    @staticmethod
    def action_reward(action: jnp.array) -> jnp.array:
        """ Computes the reward/penalty for the given action """
        return - (action ** 2).sum(-1)

    def state_reward(self, obs: jnp.array, next_obs: jnp.array) -> jnp.array:
        """ Computes the reward for the given observations """
        if self.encode_angle:
            next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
        pos_diff = next_obs[..., :2] - self.goal[:2]
        theta_diff = next_obs[..., 2] - self.goal[2]
        pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
        theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        reward = self.tolerance_pos(pos_dist) + 0.5 * self.tolerance_theta(theta_dist)
        return reward

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)


class RCCarSimEnv:
    max_steps: int = 200
    _dt: float = 1/30.
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, - jnp.pi / 2.])
    _init_pose: jnp.array = jnp.array([-1.04, -1.42, jnp.pi / 2.])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = 0.05 * jnp.exp(jnp.array([-3.3170326, -3.7336411, -2.7081904,
                                                -2.7841284, -2.7067015, -1.4446207]))
    _default_car_model_params: Dict = {
        'use_blend': 0.0,
        'm': 1.3,
        'c_m_1': 1.0,
        'c_m_2': 0.2,
        'c_d': 0.5,
        'l_f': 0.3,
        'l_r': 0.3,
        'steering_limit': 0.5
    }

    def __init__(self, ctrl_cost_weight: float = 0.005, encode_angle: bool = False, use_obs_noise: bool = True,
                 car_model_params: Dict = None, seed: int = 230492394):
        self.dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)

        # initialize dynamics and observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=False)

        if car_model_params is None:
            _car_model_params = self._default_car_model_params
        else:
            _car_model_params = self._default_car_model_params
            _car_model_params.update(car_model_params)
        self._dynamics_params = CarParams(**_car_model_params)
        self._next_step_fn = jax.jit(partial(self._dynamics_model.next_step, params=self._dynamics_params))

        self.use_obs_noise = use_obs_noise

        # initialize reward model
        self._reward_model = RCCarEnvReward(goal=self._goal,
                                            ctrl_cost_weight=ctrl_cost_weight,
                                            encode_angle=self.encode_angle)

        # initialize time and state
        self._time: int = 0
        self._state: jnp.array = jnp.zeros(self.dim_state)

    def reset(self, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.array:
        """ Resets the environment to a random initial state close to the initial pose """
        rng_key = self.rds_key if rng_key is None else rng_key

        # sample random initial state
        key_pos, key_theta, key_vel, key_obs = jax.random.split(rng_key, 4)
        init_pos = self._init_pose[:2] + jax.random.uniform(key_pos, shape=(2,), minval=-0.10, maxval=0.10)
        init_theta = self._init_pose[2:] + \
                     jax.random.uniform(key_pos, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi)
        init_vel = jnp.zeros((3,)) + jnp.array([0.005, 0.005, 0.02]) * jax.random.normal(key_vel, shape=(3,))
        init_state = jnp.concatenate([init_pos, init_theta, init_vel])

        self._state = init_state
        self._time = 0
        return self._state_to_obs(self._state, rng_key=key_obs)

    def step(self, action: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None) \
            -> Tuple[jnp.array, float, bool, Dict[str, Any]]:
        """ Performs one step in the environment """

        assert action.shape[-1:] == self.dim_action
        #xassert jnp.all(-1 <= action) and jnp.all(action <= 1), "action must be in [-1, 1]"
        rng_key = self.rds_key if rng_key is None else rng_key

        # compute next state
        self._state = self._next_step_fn(self._state, action)
        self._time += 1
        obs = self._state_to_obs(self._state, rng_key=rng_key)

        # compute reward
        reward = self._reward_model.forward(obs=None, action=action, next_obs=obs)

        # check if done
        done = self._time >= self.max_steps

        # return observation, reward, done, info
        return obs, reward, done, {'time': self._time, 'state': self._state,
                                   'reward': reward}

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
            obs = encode_angles(obs, self._angle_idx)
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
    ENCODE_ANGLE = False
    env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                      use_obs_noise=True)

    t_start = time.time()

    s = env.reset()
    traj = [s]
    rewards = []
    for i in range(200):
        t = i / 30.
        a = jnp.array([- 1 * jnp.cos(1.0 * t), 0.8 / (t+1)])
        s, r, _, _ = env.step(a)
        traj.append(s)
        rewards.append(r)

    duration = time.time() - t_start
    print(f'Duration of trajectory sim {duration} sec')
    traj = jnp.stack(traj)

    plot_rc_trajectory(traj, encode_angle=ENCODE_ANGLE)
