import time
from functools import partial
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from sim_transfer.sims.dynamics_models import RaceCar, CarParams
from sim_transfer.sims.tolerance_reward import ToleranceReward
from sim_transfer.sims.util import encode_angles, decode_angles, plot_rc_trajectory


class RCCarEnvReward:
    _angle_idx: int = 2
    dim_action: Tuple[int] = (2,)

    def __init__(self, goal: jnp.array, encode_angle: bool = False, ctrl_cost_weight: float = 0.005,
                 bound: float = 0.1, ):
        self.goal = goal
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle = encode_angle
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound), margin=10 * bound, value_at_margin=0.1,
                                                sigmoid='long_tail')

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
        total_dist = jnp.sqrt(pos_dist ** 2 + theta_dist ** 2)
        reward = self.tolerance_reward(total_dist)
        return reward

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)


class RCCarSimEnv:
    max_steps: int = 200
    _dt: float = 1 / 30.
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, - jnp.pi / 2.])
    _init_pose: jnp.array = jnp.array([1.04, -1.42, jnp.pi])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = 0.05 * jnp.exp(jnp.array([-3.3170326, -3.7336411, -2.7081904,
                                                           -2.7841284, -2.7067015, -1.4446207]))
    _default_car_model_params_bicycle: Dict = {
        'use_blend': 0.0,
        'm': 1.65,
        'l_f': 0.13,
        'l_r': 0.17,
        'angle_offset': -0.01929582,
        'b_f': 2.58,
        'b_r': 3.39,
        'blend_ratio_lb': 0.01,
        'blend_ratio_ub': 0.01,
        'c_d': 0.0,
        'c_f': 1.4,
        'c_m_1': 10.431917,
        'c_m_2': 1.0,
        'c_r': 1.27,
        'd_f': 0.4,
        'd_r': 0.48,
        'i_com': 0.85,
        'steering_limit': 0.45
    }

    _default_car_model_params_blend: Dict = {
        'use_blend': 1.0,
        'm': 1.65,
        'l_f': 0.13,
        'l_r': 0.17,
        'angle_offset': 0.00731506,
        'b_f': 2.5134025,
        'b_r': 3.8303657,
        'blend_ratio_lb': 0.002,
        'blend_ratio_ub': 0.015,
        'c_d': -6.9619144e-37,
        'c_f': 1.5,
        'c_m_1': 10.93334,
        'c_m_2': 1.08,
        'c_r': 1.28,
        'd_f': 0.35,
        'd_r': 0.463,
        'i_com': 0.1,
        'steering_limit': 0.55,
    }

    def __init__(self, ctrl_cost_weight: float = 0.005, encode_angle: bool = False, use_obs_noise: bool = True,
                 use_tire_model: bool = False, action_delay: float = 0.0, car_model_params: Dict = None,
                 max_throttle: float = 0.5, seed: int = 230492394):
        """
        Race car simulator environment

        Args:
            ctrl_cost_weight: weight of the control penalty
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_obs_noise: whether to use observation noise
            use_tire_model: whether to use the (high-fidelity) tire model, if False just uses a kinematic bicycle model
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            car_model_params: dictionary of car model parameters that overwrite the default values
            seed: random number generator seed
        """
        self.dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)
        self.max_throttle = jnp.clip(max_throttle, 0.0, 1.0)

        # initialize dynamics and observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=False)

        self.use_tire_model = use_tire_model
        if use_tire_model:
            self._default_car_model_params = self._default_car_model_params_blend
        else:
            self._default_car_model_params = self._default_car_model_params_bicycle

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

        # set up action delay
        assert action_delay >= 0.0, "Action delay must be non-negative"
        self.action_delay = action_delay
        if action_delay % self._dt == 0.0:
            self._act_delay_interpolation_weights = jnp.array([1.0, 0.0])
        else:
            # if action delay is not a multiple of dt, compute weights to interpolate
            # between temporally closest actions
            weight_first = (action_delay % self._dt) / self._dt
            self._act_delay_interpolation_weights = jnp.array([weight_first, 1.0 - weight_first])
        action_delay_buffer_size = int(jnp.ceil(action_delay / self._dt)) + 1
        self._action_buffer = jnp.zeros((action_delay_buffer_size, self.dim_action[0]))

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
        """ Performs one step in the environment

        Args:
            action: array of size (2,) with [steering, throttle]
            rng_key: rng key for the observation noise (optional)
        """

        assert action.shape[-1:] == self.dim_action
        action = np.clip(action, -1.0, 1.0)
        action = action.at[0].set(self.max_throttle * action[0])
        # assert jnp.all(-1 <= action) and jnp.all(action <= 1), "action must be in [-1, 1]"
        rng_key = self.rds_key if rng_key is None else rng_key

        if self.action_delay > 0.0:
            # pushes action to action buffer and pops the oldest action
            # computes delayed action as a linear interpolation between the relevant actions in the past
            action = self._get_delayed_action(action)

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

    def _get_delayed_action(self, action: jnp.array) -> jnp.array:
        # push action to action buffer
        self._action_buffer = jnp.concatenate([self._action_buffer[1:], action[None, :]], axis=0)

        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        delayed_action = jnp.sum(self._action_buffer[:2] * self._act_delay_interpolation_weights[:, None], axis=0)
        assert delayed_action.shape == self.dim_action
        return delayed_action

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
                      action_delay=0.00,
                      use_tire_model=True,
                      use_obs_noise=True)

    t_start = time.time()
    s = env.reset()
    traj = [s]
    rewards = []
    actions = []
    for i in range(50):
        t = i / 30.
        a = jnp.array([- 1 * jnp.cos(1.0 * t), 0.8 / (t + 1)])
        s, r, _, _ = env.step(a)
        traj.append(s)
        actions.append(a)
        rewards.append(r)

    duration = time.time() - t_start
    print(f'Duration of trajectory sim {duration} sec')
    traj = jnp.stack(traj)
    actions = jnp.stack(actions)

    plot_rc_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)
