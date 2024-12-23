import matplotlib.pyplot as plt
import numpy as np

from experiments.data_provider import provide_data_and_sim


def provide_names(state_dim: int):
    "Provide names of race car state variables."
    state_vars = ['x', 'y', 'cos(phi)', 'sin(phi)', 'vx', 'vy', 'omega']
    num_actions = (state_dim - 7) // 2
    actions_vars = []
    for i in range(num_actions):
        actions_vars.append(f'steer_{i}')
        actions_vars.append(f'throttle_{i}')
    return state_vars + actions_vars


def get_data(num_offline_collected_transitions: int, iid: bool = True):
    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
        data_source='real_racecar_new_actionstack',
        data_spec={'num_samples_train': num_offline_collected_transitions,
                   'use_hf_sim': True,
                   'sampling': 'iid' if iid else 'consecutive'
                   })

    return x_train, y_train


def plot_data(data: np.ndarray, normalize: bool = False, title: str = None):
    num_dims = data.shape[1]
    if normalize:
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    fig, ax = plt.subplots(num_dims, 1, figsize=(8, 2 * num_dims))
    data_names = provide_names(data.shape[1])
    for i in range(num_dims):
        ax[i].hist(data[:, i], bins=100)
        ax[i].set_title(data_names[i])
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    num_offline_collected_transitions = 20_000
    x_train, y_train = get_data(num_offline_collected_transitions)
    # plot_data(x_train, normalize=False, title='Intput data')
    plot_data(y_train, normalize=False, title='Output data')
