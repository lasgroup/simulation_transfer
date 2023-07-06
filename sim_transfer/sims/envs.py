import jax.numpy as jnp
import jax
import time

from functools import partial
from sim_transfer.sims.dynamics_models import RaceCar, CarParams
from typing import Union, Dict, Any, Callable, Tuple, Optional, List

def plot_trajectory(traj: jnp.array, show: bool = True):
    import matplotlib.pyplot as plt
    scale_factor = 1.5
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(scale_factor * 12, scale_factor * 8))
    size = 3
    axes[0][0].set_xlim(-size, size)
    axes[0][0].set_ylim(-size, size)
    axes[0][0].scatter(0, 0)
    # axes[0][0].plot(traj[:, 0], traj[:, 1])
    axes[0][0].set_title('x-y')
    # Plot the velocity of the car as vectors
    total_vel = jnp.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2)
    axes[0][0].quiver(traj[0:-1:3, 0], traj[0:-1:3, 1], traj[0:-1:3, 3], traj[0:-1:3, 4],
                      total_vel[0:-1:3], cmap='jet', scale=20,
                      headlength=2, headaxislength=2, headwidth=2, linewidth=0.2)

    t = jnp.arange(traj.shape[0]) / 30.
    # theta
    axes[0][1].plot(t, traj[:, 2])
    axes[0][1].set_xlabel('time')
    axes[0][1].set_ylabel('theta')
    axes[0][1].set_title('theta')

    # angular velocity
    axes[0][2].plot(t, traj[:, -1])
    axes[0][2].set_xlabel('time')
    axes[0][2].set_ylabel('angular velocity')
    axes[0][2].set_title('angular velocity')

    axes[1][0].plot(t, total_vel)
    axes[1][0].set_xlabel('time')
    axes[1][0].set_ylabel('total velocity')
    axes[1][0].set_title('velocity')

    # vel x
    axes[1][1].plot(t, traj[:, 3])
    axes[1][1].set_xlabel('time')
    axes[1][1].set_ylabel('velocity x')
    axes[1][1].set_title('velocity x')

    axes[1][2].plot(t, traj[:, 4])
    axes[1][2].set_xlabel('time')
    axes[1][2].set_ylabel('velocity y')
    axes[1][2].set_title('velocity y')

    fig.tight_layout()
    if show:
        fig.show()
    return fig, axes


class RCCarSimEnv:
    max_steps: int = 200
    _dt: float = 1/30.
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, - jnp.pi / 2.])
    _init_pose: jnp.array = jnp.array([-1.04, -1.42, jnp.pi / 2.])
    _angle_idx: int = 2

    def __init__(self, ctrl_cost_weight: float = 0.005, encode_angle: bool = False, rk_integrator: bool = True,
                 seed: int = 230492394):
        self.dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)

        # initialize dynamics and observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=self.encode_angle)
        self._dynamics_params = CarParams(use_blend=0.0)  # TODO allow setting the params
        self._next_step_fn = jax.jit(partial(self._dynamics_model.next_step, params=self._dynamics_params))

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

        if self.encode_angle:
            init_state = self._encode_angles(init_state)
        self._state = init_state
        self._time = 0
        return self._state

    def step(self, action: jnp.array) -> Tuple[jnp.array, float, bool, Dict[str, Any]]:
        """ Performs one step in the environment """
        assert action.shape[-1:] == self.dim_action

        # compute next state
        self._state = self._next_step_fn(self._state, action)
        self._time += 1
        obs = self._state

        # compute reward
        reward = 10.   # TODO

        # check if done
        done = self._time >= self.max_steps

        # return observation, reward, done, info
        return obs, reward, done, {}

    @property
    def rds_key(self) -> jax.random.PRNGKey:
        self._rds_key, key = jax.random.split(self._rds_key)
        return key

    def _encode_angles(self, state: jnp.array) -> jnp.array:
        """ Encodes the angle (theta) as sin(theta) and cos(theta) """
        assert state.shape[-1] == 6
        theta = state[..., self._angle_idx:self._angle_idx+1]
        state_encoded = jnp.concatenate([state[..., :self._angle_idx], jnp.sin(theta), jnp.cos(theta),
                                         state[..., self._angle_idx+1:]], axis=-1)
        assert state_encoded.shape[-1] == 7
        return state_encoded

    def _decode_angles(self, state: jnp.array) -> jnp.array:
        """ Decodes the angle (theta) from sin(theta) and cos(theta)"""
        assert state.shape[-1] == 7
        theta = jnp.arctan2(state[..., self._angle_idx:self._angle_idx+1],
                            state[..., self._angle_idx+1:self._angle_idx+2])
        state_decoded = jnp.concatenate([state[..., :self._angle_idx], theta, state[..., self._angle_idx+2:]], axis=-1)
        assert state_decoded.shape[-1] == 6
        return state_decoded



if __name__ == '__main__':
    env = RCCarSimEnv(encode_angle=False)

    t_start = time.time()

    s = env.reset()
    traj = [s]
    for i in range(200):
        t = i / 30.
        a = jnp.array([jnp.sin(t), 0.2 / (t+1)])
        s, _, _, _ = env.step(a)
        print(a)
        traj.append(s)

    duration = time.time() - t_start
    print(f'Duration of trajectory sim {duration} sec')
    traj = jnp.stack(traj)

    plot_trajectory(traj)
