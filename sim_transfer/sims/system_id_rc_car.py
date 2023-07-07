import pandas as pd
import os
import glob
import jax.numpy as jnp
import jax
import optax
import numpy as np
from functools import partial
from sim_transfer.sims.dynamics_models import RaceCar, CarParams
from sim_transfer.sims.util import angle_diff, plot_rc_trajectory
from matplotlib import pyplot as plt

import tensorflow_probability.substrates.jax.distributions as tfd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
BATCH_SIZE = 64
NUM_STEPS_AHEAD = 3
REAL_DATA = False

def load_recordings(recordings_dir: str):
    dfs = []
    for path in glob.glob(os.path.join(recordings_dir, '*sampled.csv')):
        df = pd.read_csv(path)
        df.columns = [c[1:] for c in df.columns]
        dfs.append(df)
    return dfs

def get_tajectory_windows(arr: jnp.array, window_size: int = 10) -> jnp.array:
    """Sliding window over an array along the first axis."""
    arr_strided = jnp.stack([arr[i:(-window_size + i)] for i in range(window_size)], axis=-2)
    assert arr_strided.shape == (arr.shape[0] - window_size, window_size, arr.shape[-1])
    return jnp.array(arr_strided)

def prepare_data(df: pd.DataFrame, window_size=10):
    u = df[['steer', 'throttle']].to_numpy()
    x = df[['pos x', 'pos y', 'theta', 's vel x', 's vel y', 's omega']].to_numpy()

    # project theta into [-\pi, \pi]
    x[:, 2] = (x[:, 2] + jnp.pi) % (2 * jnp.pi) - jnp.pi

    x_strides = get_tajectory_windows(x, window_size)
    u_strides = get_tajectory_windows(u, window_size)
    return x_strides, u_strides

recordings_dir = os.path.join(DATA_DIR, 'recordings_rc_car_v0' if REAL_DATA else 'simulated_rc_car_v0')
num_train_traj = 2 if REAL_DATA else 7
recording_dfs = load_recordings(recordings_dir)
datasets_train = list(map(partial(prepare_data, window_size=11), recording_dfs[:num_train_traj]))
datasets_test = list(map(partial(prepare_data, window_size=61), recording_dfs[num_train_traj:]))

x_train, u_train = map(lambda x: jnp.concatenate(x, axis=0), zip(*datasets_train))
x_test, u_test = map(lambda x: jnp.concatenate(x, axis=0), zip(*datasets_test))

plot_rc_trajectory(x_test[0], show=True)

dynamics = RaceCar(dt=1 / 30., encode_angle=False, rk_integrator=True)
step_vmap = jax.vmap(dynamics.next_step, in_axes=(0, 0, None), out_axes=0)

params_car_model = {
    #'m': jnp.array(0.05),
    'i_com': jnp.array(27.8e-6),
    'l_f': jnp.array(0.03),  # length to the front from COM  #TODO 0.3
    'l_r': jnp.array(0.035),  # length to the back from COM  #TODO 0.3
    #'g': jnp.array(9.81),
    'd_f': jnp.array(0.02),
    'c_f': jnp.array(1.2),
    'b_f': jnp.array(2.58),
    'd_r': jnp.array(0.017),
    'c_r': jnp.array(1.27),
    'b_r': jnp.array(3.39),
    'c_m_1': jnp.array(0.2),
    'c_m_2': jnp.array(0.05),
    'c_d': jnp.array(0.052),
    'steering_limit': jnp.array(0.35),
    #'use_blend': jnp.array(0.0),
}

params = {'car_model': params_car_model,
          'noise_log_std': -1. * jnp.ones((NUM_STEPS_AHEAD, 6))}

optim = optax.adam(1e-3)
opt_state = optim.init(params)


def simulate_traj(x0: jnp.array, u_traj, params, num_steps: int) -> jnp.array:
    pred_traj = [x0]
    x = x0
    for i in range(num_steps):
        x_pred = step_vmap(x, u_traj[..., i, :], CarParams(**params['car_model'], m=1.3, g=9.81, use_blend=0.0))
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
    diff = jnp.concatenate([diff[..., :angle_idx], theta_diff[..., None], diff[..., angle_idx+1:]], axis=-1)
    assert diff.shape == traj1.shape
    return diff


def loss_fn(params, x_strided, u_strided, num_steps_ahead: int = 3,
            exclude_ang_vel: bool = False):
    assert x_strided.shape[-2] > num_steps_ahead

    pred_traj = simulate_traj(x_strided[..., 0, :], u_strided, params, num_steps_ahead)
    pred_traj = pred_traj[..., 1:, :]  # remove first state (which is the initial state)

    # compute diff between predicted and real trajectory
    real_traj = x_strided[..., 1:1+num_steps_ahead, :]
    diff = trajecory_diff(real_traj, pred_traj)

    pred_dist = tfd.Normal(jnp.zeros_like(params['noise_log_std']), jnp.exp(params['noise_log_std']))
    if exclude_ang_vel:
        loss = - jnp.mean(pred_dist.log_prob(diff)[..., :5])
    else:
        loss = - jnp.mean(pred_dist.log_prob(diff))
    return loss

def plot_trajectory_comparison(real_traj, sim_traj):
    assert real_traj.shape == sim_traj.shape and real_traj.shape[-1] == 6 and real_traj.ndim == 2
    fig, axes = plt.subplots(ncols=2)
    #ax.scatter(sim_traj[0, 0], sim_traj[0, 1], color='green')
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
    idx_batch = jax.random.choice(key, x_train.shape[0], shape=(BATCH_SIZE,))
    x_batch, u_batch = x_train[idx_batch], u_train[idx_batch]
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, u_batch)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


def eval(params, x_eval, u_eval, log_plots: bool = False):
    traj_pred = simulate_traj(x_eval[..., 0, :], u_eval, params, num_steps=60)
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
        plots = {
            'trajectory_comparison_1': plot_trajectory_comparison(x_eval[100], traj_pred[100]),
            'trajectory_comparison_500': plot_trajectory_comparison(x_eval[500], traj_pred[500]),
            'trajectory_comparison_2000': plot_trajectory_comparison(x_eval[2000], traj_pred[2000])
        }
        return {**metrics, **plots}
    else:
        return metrics


key = jax.random.PRNGKey(234234)

import wandb
run = wandb.init(
  project="system-id-rccar",
  entity="jonasrothfuss"
)

for i in range(20000):
    key, subkey = jax.random.split(key)
    loss, params, opt_state = step(params, opt_state, subkey)

    if i % 1000 == 0:
        loss_test = loss_fn(params, x_test, u_test)
        metrics_eval = eval(params, x_test, u_test, log_plots=True)
        wandb.log({'iter': i,  'loss': loss, 'loss_test': loss_test, **metrics_eval})
        print(f'Iter {i}, loss: {loss}, test loss: {loss_test}')


from pprint import pprint
pprint(params)


