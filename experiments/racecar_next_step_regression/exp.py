import argparse
from typing import Tuple

import jax
import jax.numpy as jnp

from experiments.data_provider import _RACECAR_NOISE_STD_ENCODED
from sim_transfer.models import BNN_SVGD, BNN_FSVGD_SimPrior
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.simulators import RaceCarSim, PredictStateChangeWrapper


def experiment(num_train_traj: int,
               use_sim_prior: int):
    env = RCCarSimEnv(encode_angle=True, action_delay=0.00,
                      use_tire_model=True, use_obs_noise=True)

    NUM_TRAJECTORIES = 50

    """ Collect or load trajectories/rollouts """

    def collect_traj(env, action_fn: callable, n_steps: int = 200) -> Tuple[jnp.ndarray, jnp.ndarray]:
        s = env.reset()
        traj = [s]
        rewards = []
        actions = []
        for i in range(n_steps):
            t = i / 30.
            a = action_fn(t)
            s, r, _, _ = env.step(a)
            traj.append(s)
            actions.append(a)
            rewards.append(r)

        traj = jnp.stack(traj)
        actions = jnp.stack(actions)
        return traj, actions

    def collect_rollouts(num_rollouts: int):
        key = jax.random.PRNGKey(234234)
        rollouts = []
        for _ in range(num_rollouts):
            key1, key2, key3, key = jax.random.split(key, 4)
            freq = jax.random.uniform(key1, shape=(), minval=0.7, maxval=1.5)
            phase = jax.random.uniform(key2, shape=(), minval=0.0, maxval=2 * jnp.pi)
            dec = jax.random.uniform(key3, shape=(), minval=0.7, maxval=1.3)
            action_fn = lambda t: jnp.array([jnp.cos(freq * t + phase),
                                             0.8 / (dec * t + 1)])

            traj, actions = collect_traj(env, action_fn, n_steps=200)
            # plot_rc_trajectory(traj, actions, encode_angle=True)
            rollouts.append((traj, actions))
        return rollouts

    print(f'Collecting {NUM_TRAJECTORIES} rollouts.')
    rollouts = collect_rollouts(NUM_TRAJECTORIES)

    """ Convert trajectories to regresion data """

    def convert_traj_to_data(state_traj, action_traj):
        x_data = jnp.concatenate([state_traj[:-1], action_traj], axis=-1)
        # Here target is difference and not absolute state
        y_data = state_traj[1:] - state_traj[:-1]
        return x_data, y_data

    data = list(map(lambda x: convert_traj_to_data(*x), rollouts))

    x_train, y_train = list(map(lambda l: jnp.concatenate(l, axis=0), zip(*data[:num_train_traj])))
    x_test, y_test = list(map(lambda l: jnp.concatenate(l, axis=0), zip(*data[num_train_traj:])))

    print('Num trajectories for training:', args.num_train_traj)
    print('Train data shapes:', x_train.shape, y_train.shape)
    print('Test data shapes:', x_test.shape, y_test.shape)

    """ Train a neural network """

    _sim = RaceCarSim(encode_angle=True, use_blend=True)
    sim = PredictStateChangeWrapper(_sim)

    standard_model_params = {
        'input_size': x_train.shape[-1],
        'output_size': y_train.shape[-1],
        'rng_key': jax.random.PRNGKey(234234345),
        'normalization_stats': sim.normalization_stats,
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': False,
        'likelihood_exponent': 0.5,
        'hidden_layer_sizes': [64, 64, 64],
        'data_batch_size': 32,

    }

    print('Using sim prior:', bool(use_sim_prior))

    if use_sim_prior:
        bnn = BNN_FSVGD_SimPrior(domain=sim.domain,
                                 function_sim=sim,
                                 num_measurement_points=16,
                                 num_f_samples=512,
                                 score_estimator='gp',
                                 **standard_model_params,
                                 num_train_steps=400000
                                 )
    else:
        bnn = BNN_SVGD(**standard_model_params,
                       bandwidth_svgd=1.0,
                       num_train_steps=20000)

    bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test)


def main(args):
    experiment(
        num_train_traj=args.num_train_traj,
        use_sim_prior=args.use_sim_prior,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_traj', type=int, default=2)
    parser.add_argument('--use_sim_prior', type=int, default=1)
    args = parser.parse_args()
