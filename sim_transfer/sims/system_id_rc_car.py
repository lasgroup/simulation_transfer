import pandas as pd
import os
import glob
import jax.numpy as jnp
import jax
import optax
from functools import partial
from sim_transfer.sims.dynamics_models import RaceCar, CarParams

import tensorflow_probability.substrates.jax.distributions as tfd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
ENCODE_ANGLE = True
BATCH_SIZE = 64

def load_recordings(recordings_dir: str):
    dfs = []
    for path in glob.glob(os.path.join(recordings_dir, '*sampled.csv')):
        df = pd.read_csv(path)
        df.columns = [c[1:] for c in df.columns]
        dfs.append(df)
    return dfs

def prepare_data(df: pd.DataFrame, encode_angle: bool = ENCODE_ANGLE):
    u = df[['steer', 'throttle']].to_numpy()
    x = df[['pos x', 'pos y', 'theta', 's vel x', 's vel y', 's omega']].to_numpy()

    # encode angle to sin and cos
    if encode_angle:
        angle_idx = 2
        theta = x[:, angle_idx:angle_idx+1]
        x = jnp.concatenate([x[..., 0:angle_idx], jnp.sin(theta), jnp.cos(theta), x[..., angle_idx + 1:]], axis=-1)
    else:
        # project theta into [-\pi, \pi]
        x[:, 2] = x[:, 2] % (2 * jnp.pi) - jnp.pi

    x_next = x[1:]
    u = u[:-1]
    x = x[:-1]
    return x, u, x_next

recordings_dir = os.path.join(DATA_DIR, 'recordings_rc_car_v0')
datasets = list(map(partial(prepare_data, encode_angle=ENCODE_ANGLE), load_recordings(recordings_dir))) # TODO use all recordings


x_train, u_train, x_next_train = map(lambda x: jnp.concatenate(x, axis=0), zip(datasets[0], datasets[1]))
x_test, u_test, x_next_test = datasets[2]

dynamics = RaceCar(dt=1 / 30., encode_angle=ENCODE_ANGLE, rk_integrator=True)

step_jitted = jax.jit(jax.vmap(dynamics.next_step, in_axes=(0, 0, None), out_axes=0))

params_car_model = {
    'm': jnp.array(0.05),  # TODO 1.3
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
    'c_rr': jnp.array(0.003),
    'c_d': jnp.array(0.052),
    'steering_limit': jnp.array(0.35),
    #'use_blend': jnp.array(0.0),
}

params = {'car_model': params_car_model,
          'noise_log_std': -1. * jnp.ones(7 if ENCODE_ANGLE else 6)}

optim = optax.adam(1e-3)
opt_state = optim.init(params)

def loss_fn(params, x, u, x_next):
    x_pred = step_jitted(x, u, CarParams(**params['car_model']))
    pred_dist = tfd.MultivariateNormalDiag(x_pred, jnp.exp(params['noise_log_std']))
    loss = - jnp.mean(pred_dist.log_prob(x_next))
    return loss

@jax.jit
def step(params, opt_state, key: jax.random.PRNGKey):
    idx_batch = jax.random.choice(key, x_train.shape[0], shape=(BATCH_SIZE,))
    x_batch, u_batch, x_next_batch = x_train[idx_batch], u_train[idx_batch], x_next_train[idx_batch]
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, u_batch, x_next_batch)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

key = jax.random.PRNGKey(234234)

for i in range(10000):
    key, subkey = jax.random.split(key)
    loss, params, opt_state = step(params, opt_state, subkey)

    if i % 1000 == 0:
        loss_test = loss_fn(params, x_test, u_test, x_next_test)
        print(f'Iter {i}, loss: {loss}, test loss: {loss_test}')

from pprint import pprint
pprint(params)


