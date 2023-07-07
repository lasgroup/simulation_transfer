import pandas as pd

from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.util import plot_rc_trajectory
import time
import jax
import os
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax as tfp
from matplotlib import pyplot as plt


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
sims_dir = os.path.join(DATA_DIR, 'simulated_rc_car_v0')
os.makedirs(sims_dir, exist_ok=True)

ENCODE_ANGLE = False
env = RCCarSimEnv(encode_angle=ENCODE_ANGLE, use_obs_noise=True)

t_start = time.time()

key = jax.random.PRNGKey(2342)

def generate_action_trajectory(key, num_steps=100, length_scale: float = 0.5):
    t = jnp.arange(num_steps) / 30.
    K = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=length_scale).matrix(t[:, None], t[:, None]) \
        + 5e-4 * jnp.eye(num_steps)
    dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=K)
    key1, key2 = jax.random.split(key)
    s1 = 0.5 * jnp.sin(t) + jnp.clip(0.7 * dist.sample(seed=key1), -1, 1)
    s2 = jnp.clip(0.5 + 0.5 * dist.sample(seed=key2), -1, 1)
    return jnp.stack([s1, s2], axis=-1)

def simulate_traj(s0, actions, env):
    traj = [s0]
    rewards = []
    for a in actions:
        s, _, _, _ = env.step(a)
        traj.append(s)
    return jnp.stack(traj, axis=0)


# t = jnp.arange(actions.shape[0]) / 30.
# plt.plot(t, actions[:, 0])
# plt.plot(t, actions[:, 1])
# plt.show()


for i, key in enumerate(jax.random.split(key, 10)):
    key1, key2 = jax.random.split(key)
    actions = generate_action_trajectory(key1, length_scale=1., num_steps=600)
    s0 = env.reset(key2)
    traj = simulate_traj(s0, actions, env, )

    df = pd.DataFrame({
        ' steer': actions[:, 0],
        ' throttle': actions[:, 1],
        ' pos x': traj[:-1, 0],
        ' pos y': traj[:-1, 1],
        ' theta': traj[:-1, 2],
        ' s vel x': traj[:-1, 3],
        ' s vel y': traj[:-1, 4],
        ' s omega': traj[:-1, 5],
    }
    )
    df.to_csv(os.path.join(sims_dir, f'sim{i}_sampled.csv'), index=False)

    plot_rc_trajectory(traj, encode_angle=ENCODE_ANGLE)
