import pickle

import pandas as pd
import os
import jax.numpy as jnp
import jax
import optax
import numpy as np
import argparse
import glob
from functools import partial

from experiments.util import get_trajectory_windows
from sim_transfer.sims.dynamics_models import RaceCar, CarParams
from sim_transfer.sims.util import angle_diff
from matplotlib import pyplot as plt
from brax.training.types import Transition

import tensorflow_probability.substrates.jax.distributions as tfd


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--batch_size', type=int, default=64)
arg_parser.add_argument('--num_steps_ahead', type=int, default=3)
arg_parser.add_argument('--action_delay', type=int, default=3)
arg_parser.add_argument('--real_data', type=bool, default=True)
arg_parser.add_argument('--encode_angle', type=bool, default=True)
arg_parser.add_argument('--use_blend', type=float, default=1.0)
arg_parser.add_argument('--data_source', type=str, default='recordings_rc_car_v4')
arg_parser.add_argument('--seed', type=int, default=234234)
args = arg_parser.parse_args()

key_init, key_train, key_data = jax.random.split(jax.random.PRNGKey(args.seed), 3)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        f'data/{args.data_source}')


def rotate_vector(v, theta):
    v_x, v_y = v[..., 0], v[..., 1]
    rot_x = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
    rot_y = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
    return jnp.concatenate([jnp.atleast_2d(rot_x), jnp.atleast_2d(rot_y)], axis=0).T


def prepare_data(transitions: Transition, window_size=10, encode_angles: bool = False, action_delay: int = 0):
    assert 0 <= action_delay <= 3, "Only recorded the last 3 actions and the current action"
    if action_delay == 0:
        u = transitions.action
    elif action_delay == 1:
        u = transitions.observation[:, -2:]
    else:
        u = transitions.observation[:, -2 * action_delay: -2 * (action_delay - 1)]

    x = transitions.observation[:, :6]
    # project theta into [-\pi, \pi]
    x[:, 2] = (x[:, 2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    if encode_angles:
        def angle_encode(obs):
            theta = obs[2]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            encoded_obs = jnp.array([obs[0], obs[1], sin_theta, cos_theta, obs[3], obs[4], obs[5]])
            return encoded_obs

        x = jax.vmap(angle_encode)(x)
    x_strides = get_trajectory_windows(x, window_size)
    u_strides = get_trajectory_windows(u, window_size)
    return x_strides, u_strides


def load_transitions(file_dir):
    file_names = glob.glob(os.path.join(file_dir, '*.pickle'))
    transitions = []
    for fn in file_names:
        with open(fn, 'rb') as f:
            transitions.append(pickle.load(f))
    return transitions

if args.data_source == 'recordings_rc_car_v3':
    num_train_traj = 4
else:
    num_train_traj = 10 if args.real_data else 7
transitions = load_transitions(DATA_DIR)
datasets = list(map(partial(prepare_data, window_size=11, encode_angles=args.encode_angle, action_delay=args.action_delay),
                            transitions))

# shuffle data
datasets = [datasets[i] for i in jax.random.permutation(key_data, jnp.arange(len(datasets)))]

# split into train and test
datasets_train = datasets[:num_train_traj]
datasets_test = datasets[num_train_traj:]
x_train, u_train = map(lambda x: jnp.concatenate(x, axis=0), zip(*datasets_train))
x_test, u_test = map(lambda x: jnp.concatenate(x, axis=0), zip(*datasets_test))

dynamics = RaceCar(dt=1 / 30., encode_angle=args.encode_angle, rk_integrator=True)
step_vmap = jax.vmap(dynamics.next_step, in_axes=(0, 0, None), out_axes=0)


k1, k2, k3, k4 = jax.random.split(key_init, 4)
params_car_model = {
    'i_com': jnp.array(27.8e-6),
    'd_f': jnp.array(0.02),
    'c_f': jnp.array(1.2),
    'b_f': jnp.array(2.58),
    'd_r': jnp.array(0.017),
    'c_r': jnp.array(1.27),
    'b_r': jax.random.uniform(k4, minval=2, maxval=8),
    'c_m_1': jax.random.uniform(k1, minval=4, maxval=20),
    'c_m_2': jax.random.uniform(k2, minval=0.5, maxval=2.),
    'c_d': jnp.array(0.52),
    'steering_limit': jax.random.uniform(k3, minval=0.3, maxval=0.8),
    'blend_ratio_ub': jnp.array([0.5477225575]),
    'blend_ratio_lb': jnp.array([0.4472135955]),
    'angle_offset': jnp.array([0.0]),
    # 'use_blend': jnp.array(0.0),
}

params = {'car_model': params_car_model,
          'noise_log_std': -1. * jnp.ones((args.num_steps_ahead, 7 if args.encode_angle else 6))}

optim = optax.adam(1e-3)
opt_state = optim.init(params)


def simulate_traj(x0: jnp.array, u_traj, params, num_steps: int) -> jnp.array:
    pred_traj = [x0]
    x = x0
    for i in range(num_steps):
        x_pred = step_vmap(x, u_traj[..., i, :], CarParams(**params['car_model'], m=1.65, g=9.81,
                                                           use_blend=args.use_blend,
                                                           l_f=0.13, l_r=0.17))
        pred_traj.append(x_pred)
        x = x_pred
    pred_traj = jnp.stack(pred_traj, axis=-2)
    assert pred_traj.shape[-2:] == (num_steps + 1, x0.shape[-1])
    return pred_traj


def trajecory_diff(traj1: jnp.array, traj2: jnp.array, angle_idx: int = 2) -> jnp.array:
    """Compute the difference between two trajectories. Accounts for angles on the circle."""
    assert traj1.shape == traj2.shape
    # compute diff between predicted and real trajectory
    diff = traj1 - traj2

    # special treatment for theta (i.e. shortest distance between angles on the circle)
    theta_diff = angle_diff(traj1[..., angle_idx], traj2[..., angle_idx])
    diff = jnp.concatenate([diff[..., :angle_idx], theta_diff[..., None], diff[..., angle_idx + 1:]], axis=-1)
    assert diff.shape == traj1.shape
    return diff


def loss_fn(params, x_strided, u_strided, num_steps_ahead: int = 3,
            exclude_ang_vel: bool = False):
    assert x_strided.shape[-2] > num_steps_ahead

    pred_traj = simulate_traj(x_strided[..., 0, :], u_strided, params, num_steps_ahead)
    pred_traj = pred_traj[..., 1:, :]  # remove first state (which is the initial state)

    # compute diff between predicted and real trajectory
    real_traj = x_strided[..., 1:1 + num_steps_ahead, :]
    diff = trajecory_diff(real_traj, pred_traj)

    pred_dist = tfd.Normal(jnp.zeros_like(params['noise_log_std']), jnp.exp(params['noise_log_std']))
    if exclude_ang_vel:
        angular_velocity_idx = 6 if args.encode_angle else 5
        loss = - jnp.mean(pred_dist.log_prob(diff)[..., :angular_velocity_idx])
    else:
        loss = - jnp.mean(pred_dist.log_prob(diff))
    return loss


def plot_trajectory_comparison(real_traj, sim_traj):
    if args.encode_angle:
        def decode_angle(obs):
            sin_theta, cos_theta = obs[2], obs[3]
            theta = jnp.arctan2(sin_theta, cos_theta)
            new_obs = jnp.array([obs[0], obs[1], obs[4], theta, obs[5], obs[6]])
            return new_obs

        real_traj = jax.vmap(decode_angle)(real_traj)
        sim_traj = jax.vmap(decode_angle)(sim_traj)
    assert real_traj.shape == sim_traj.shape and real_traj.shape[-1] == 6 and real_traj.ndim == 2
    fig, axes = plt.subplots(ncols=2)
    # ax.scatter(sim_traj[0, 0], sim_traj[0, 1], color='green')
    axes[0].plot(real_traj[:, 0], real_traj[:, 1], label='real', color='green')
    axes[0].plot(sim_traj[:, 0], sim_traj[:, 1], label='sim', color='orange')
    axes[0].set_title('trajectory pos')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()

    t = np.arange(sim_traj.shape[0]) / 30.
    axes[1].plot(t, real_traj[:, 2], label='real', color='green')
    axes[1].plot(t, sim_traj[:, 2], label='sim', color='orange')
    axes[1].set_title('theta')
    axes[1].set_xlabel('time (sec)')
    axes[1].set_ylabel('theta')
    axes[1].legend()
    return fig


@jax.jit
def step(params, opt_state, key: jax.random.PRNGKey):
    idx_batch = jax.random.choice(key, x_train.shape[0], shape=(args.batch_size,))
    x_batch, u_batch = x_train[idx_batch], u_train[idx_batch]
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, u_batch)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


def eval(params, x_eval, u_eval, log_plots: bool = False):
    traj_pred = simulate_traj(x_eval[..., 0, :], u_eval, params, num_steps=10)
    diff = trajecory_diff(traj_pred, x_eval)

    pos_dist = jnp.mean(jnp.linalg.norm(diff[..., :, :2], axis=-1), axis=0)
    theta_diff = jnp.mean(jnp.abs(diff[..., 2]), axis=0)
    metrics = {
        'pos_dist_1': pos_dist[1],
        'pos_dist_5': pos_dist[5],
        'pos_dist_10': pos_dist[10],
        'pos_dist_30': pos_dist[30],
        'pos_dist_60': pos_dist[60],
        'theta_diff_1': theta_diff[1],
        'theta_diff_5': theta_diff[5],
        'theta_diff_10': theta_diff[10],
        'theta_diff_30': theta_diff[30],
        'theta_diff_60': theta_diff[60],
    }
    if log_plots:
        plots = {f'trajectory_comparison_{i}': plot_trajectory_comparison (x_eval[i], traj_pred[i])
                 for i in [1, 200, 800, 1800, 2600, 3000, 4300, 5500]}
        return {**metrics, **plots}
    else:
        return metrics


import wandb
from pprint import pprint

run = wandb.init(
    project="system-id-rccar",
    entity="jonasrothfuss",
    config=args.__dict__,
)

print('---- Config ----')
pprint(args.__dict__)
print('----------------')


for i in range(40000):
    key_train, subkey = jax.random.split(key_train)
    loss, params, opt_state = step(params, opt_state, subkey)

    if i % 1000 == 0:
        loss_test = loss_fn(params, x_test, u_test)
        metrics_eval = eval(params, x_test, u_test, log_plots=True)
        wandb.log({'iter': i, 'loss': loss, 'loss_test': loss_test, **metrics_eval})
        print(f'Iter {i}, loss: {loss}, test loss: {loss_test}')

from pprint import pprint

pprint(params)
