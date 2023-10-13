from typing import List

import chex

from experiments.data_provider import provide_data_and_sim
from experiments.offline_dataset_analysis.plot_distributions_per_dimension import plot_data

use_hf_sim = True
car_id = 2


def sample_dataset_from_simulator(num_samples_train: int,
                                  data_seed: int) -> List[chex.PRNGKey]:
    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(data_source='racecar_hf',
                                                                 data_spec={'num_samples_train': num_samples_train, },
                                                                 data_seed=data_seed)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    num_offline_collected_transitions = 20_000
    x_train, y_train = sample_dataset_from_simulator(num_samples_train=num_offline_collected_transitions,
                                                     data_seed=42)[:2]
    # plot_data(x_train, normalize=False, title='Intput data')
    plot_data(y_train, normalize=False, title='Output data')
