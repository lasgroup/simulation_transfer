from typing import Dict, Any
import jax


DEFAULTS_SINUSOIDS = {
    'obs_noise_std': 0.1,
    'x_support_mode_train': 'full',
    'param_mode': 'random',
}

def provide_data_and_sim(data_source: str, data_spec: Dict[str, Any], data_seed: int = 981648):
    # load data
    key_train, key_test = jax.random.split(jax.random.PRNGKey(data_seed), 2)
    if 'sinusoids1d' == data_source or 'sinusoids2d' == data_source:
        from sim_transfer.sims.simulators import SinusoidsSim
        sim = SinusoidsSim(input_size=1, output_size=int(data_source[-2]))

        assert {'num_samples_train'} <= set(data_spec.keys()) <= {'num_samples_train'}.union(DEFAULTS_SINUSOIDS.keys())
        x_train, y_train, x_test, y_test = sim.sample_datasets(
            rng_key=key_train,
            num_samples_train=data_spec['num_samples_train'],
            num_samples_test=1000,
            obs_noise_std=data_spec.get('obs_noise_std', DEFAULTS_SINUSOIDS['obs_noise_std']),
            x_support_mode_train=data_spec.get('x_support_mode_train', DEFAULTS_SINUSOIDS['x_support_mode_train']),
            param_mode=data_spec.get('param_mode', DEFAULTS_SINUSOIDS['param_mode'])
        )
    else:
        raise ValueError('Unknown data source %s' % data_source)

    return x_train, y_train, x_test, y_test, sim