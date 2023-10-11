from typing import List

import chex
import jax.numpy as jnp
import jax.random as jr

from experiments.data_provider import _RACECAR_NOISE_STD_ENCODED
from experiments.offline_dataset_analysis.plot_distributions_per_dimension import plot_data
from sim_transfer.sims.simulators import RaceCarSim
from sim_transfer.sims.simulators import StackedActionSimWrapper

use_hf_sim = True
car_id = 2
noise_stds = jnp.concatenate([_RACECAR_NOISE_STD_ENCODED[:2], _RACECAR_NOISE_STD_ENCODED[3:]])


def sample_dataset_from_simulator(num_samples_train: int,
                                  num_samples_test: int,
                                  key: chex.PRNGKey) -> List[chex.PRNGKey]:
    simulator = RaceCarSim(encode_angle=True, use_blend=use_hf_sim, car_id=car_id)
    sim_lf = StackedActionSimWrapper(simulator, num_stacked_actions=3, action_size=2)
    x_train, y_train, x_test, y_test = sim_lf.sample_datasets(num_samples_train=num_samples_train,
                                                              num_samples_test=num_samples_test,
                                                              obs_noise_std=noise_stds,
                                                              rng_key=key)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    num_offline_collected_transitions = 20_000
    x_train, y_train = sample_dataset_from_simulator(num_samples_train=num_offline_collected_transitions,
                                                     num_samples_test=100,
                                                     key=jr.PRNGKey(0))[:2]
    # plot_data(x_train, normalize=False, title='Intput data')
    plot_data(y_train, normalize=False, title='Output data')
