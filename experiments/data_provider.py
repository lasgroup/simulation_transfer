from typing import Dict, Any
import jax

DATASET_CONFIGS = {
    'sinusoids1d': {
        'likelihood_std': {'value': 0.1},
        'num_samples_train': {'value': 5},
    },
    'sinusoids2d': {
        'likelihood_std': {'value': 0.1},
        'num_samples_train': {'value': 5},
    },
    'pendulum': {
        'likelihood_std': {'value': 0.02},
        'num_samples_train': {'value': 20},
    },
    'pendulum_bimodal': {
        'likelihood_std': {'value': 0.02},
        'num_samples_train': {'value': 20},
    }
}

DEFAULTS_SINUSOIDS = {
    'obs_noise_std': 0.1,
    'x_support_mode_train': 'full',
    'param_mode': 'random',
}

DEFAULTS_PENDULUM = {
    'obs_noise_std': 0.02,
    'x_support_mode_train': 'full',
    'param_mode': 'random'
}

def provide_data_and_sim(data_source: str, data_spec: Dict[str, Any], data_seed: int = 845672):
    # load data
    key_train, key_test = jax.random.split(jax.random.PRNGKey(data_seed), 2)
    if data_source == 'sinusoids1d' or data_source == 'sinusoids2d':
        from sim_transfer.sims.simulators import SinusoidsSim
        DEFAULTS = DEFAULTS_SINUSOIDS
        sim_hf = sim_lf = SinusoidsSim(input_size=1, output_size=int(data_source[-2]))
        assert {'num_samples_train'} <= set(data_spec.keys()) <= {'num_samples_train'}.union(DEFAULTS_SINUSOIDS.keys())
    elif data_source == 'pendulum' or data_source == 'pendulum_hf':
        from sim_transfer.sims.simulators import PendulumSim
        DEFAULTS = DEFAULTS_PENDULUM
        if data_source == 'pendulum_hf':
            sim_hf = PendulumSim(encode_angle=True, high_fidelity=True)
            sim_lf = PendulumSim(encode_angle=True, high_fidelity=False)
        else:
            sim_hf = sim_lf = PendulumSim(encode_angle=True, high_fidelity=False)
        assert {'num_samples_train'} <= set(data_spec.keys()) <= {'num_samples_train'}.union(DEFAULTS_PENDULUM.keys())
    elif data_source == 'pendulum_bimodal' or data_source == 'pendulum_bimodal_hf':
        from sim_transfer.sims.simulators import PendulumBiModalSim
        DEFAULTS = DEFAULTS_PENDULUM
        if data_source == 'pendulum_bimodal_hf':
            sim_hf = PendulumBiModalSim(encode_angle=True, high_fidelity=True)
            sim_lf = PendulumBiModalSim(encode_angle=True, high_fidelity=False)
        else:
            sim_hf = sim_lf = PendulumBiModalSim(encode_angle=True)
        assert {'num_samples_train'} <= set(data_spec.keys()) <= {'num_samples_train'}.union(DEFAULTS_PENDULUM.keys())
    else:
        raise ValueError('Unknown data source %s' % data_source)

    x_train, y_train, x_test, y_test = sim_hf.sample_datasets(
        rng_key=key_train,
        num_samples_train=data_spec['num_samples_train'],
        num_samples_test=1000,
        obs_noise_std=data_spec.get('obs_noise_std', DEFAULTS['obs_noise_std']),
        x_support_mode_train=data_spec.get('x_support_mode_train', DEFAULTS['x_support_mode_train']),
        param_mode=data_spec.get('param_mode', DEFAULTS['param_mode'])
    )
    return x_train, y_train, x_test, y_test, sim_lf
