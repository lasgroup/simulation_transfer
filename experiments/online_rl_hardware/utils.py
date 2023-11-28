from typing import Any, NamedTuple, Dict

from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from brax.training.replay_buffers import ReplayBuffer, ReplayBufferState

from sim_transfer.rl.model_based_rl.learned_system import LearnedCarSystem
from sim_transfer.models import BNN_FSVGD_SimPrior, BNN_FSVGD, BNN_SVGD
from sim_transfer.sims.simulators import AdditiveSim, PredictStateChangeWrapper, GaussianProcessSim
from sim_transfer.sims.simulators import RaceCarSim, StackedActionSimWrapper
from sim_transfer.sims.envs import RCCarSimEnv
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from experiments.data_provider import _RACECAR_NOISE_STD_ENCODED

import pickle
import jax
import os
import jax.numpy as jnp


class ModelBasedRLConfig(NamedTuple):
    x_dim: int
    u_dim: int
    num_stacked_actions: int
    max_replay_size_true_data_buffer: int = 10 ** 4
    include_aleatoric_noise: bool = True
    car_reward_kwargs: dict = None
    sac_kwargs: dict = None
    reset_bnn: bool = True
    return_best_bnn: bool = True
    return_best_policy: bool = True
    predict_difference: bool = True
    bnn_training_test_ratio: float = 0.2
    num_stacked_actions: int = 3
    max_num_episodes: int = 100


def execute(cmd: str, verbosity: int = 0) -> None:
    if verbosity >= 2:
        print(cmd)
    os.system(cmd)


def load_data(data_load_path: str) -> Any:
    # loads the pkl file
    with open(data_load_path, 'rb') as f:
        data = pickle.load(f)
    return data


def dump_model(model: Any, model_dump_path: str) -> None:
    # dumps the model in the model_dump_path
    with open(model_dump_path, 'wb') as f:
        pickle.dump(model, f)


def dump_trajectory_summary(dump_dir: str, episode_id: int, traj_summary: Dict, verbosity: int = 1) -> None:
    assert os.path.isdir(dump_dir)
    file = os.path.join(dump_dir, f'traj_{episode_id}.pkl')
    with open(file, 'wb') as f:
        pickle.dump(traj_summary, f)
    if verbosity:
        print(f'Dumped trajectory summary to {file}')


def init_transition_buffer(config: ModelBasedRLConfig, key: jax.random.PRNGKey):
    """Setup the data buffers"""
    dummy_obs = jnp.zeros(shape=(config.x_dim + config.u_dim * config.num_stacked_actions,))
    dummy_sample = Transition(observation=dummy_obs, action=jnp.zeros(shape=(config.u_dim,)), reward=jnp.array(0.0),
                              discount=jnp.array(0.99), next_observation=dummy_obs)

    buffer = UniformSamplingQueue(max_replay_size=config.max_replay_size_true_data_buffer,
                                  dummy_data_sample=dummy_sample, sample_batch_size=1)
    buffer_state = buffer.init(key)
    return buffer, buffer_state


def add_data_to_buffer(buffer: ReplayBuffer, buffer_state: ReplayBufferState, x_data: jnp.array, y_data: jnp.array,
                       config: ModelBasedRLConfig):
    discounting = config.sac_kwargs['discounting']
    num_points = x_data.shape[0]
    assert x_data.shape == (num_points, config.x_dim + config.u_dim * (config.num_stacked_actions + 1))
    assert y_data.shape == (num_points, config.x_dim)

    observations = x_data[:, :config.x_dim + config.num_stacked_actions * config.u_dim]
    actions = x_data[:, config.x_dim + config.num_stacked_actions * config.u_dim:]
    if config.num_stacked_actions > 0:
        old_stacked_actions = observations[:, config.x_dim:]
        next_stacked_actions = jnp.concatenate([old_stacked_actions[:, config.u_dim:], actions], axis=1)
        next_observations = jnp.concatenate([y_data, next_stacked_actions], axis=1)
    else:
        next_observations = y_data
    transitions = Transition(observation=observations, action=actions, reward=jnp.zeros(shape=(num_points,)),
                             discount=jnp.ones(shape=(num_points,)) * discounting, next_observation=next_observations)

    buffer_state = buffer.insert(buffer_state, transitions)
    return buffer, buffer_state


def set_up_model_based_sac_trainer(bnn_model, data_buffer, data_buffer_state, key: jax.random.PRNGKey,
                                   config: ModelBasedRLConfig, sac_kwargs: dict = None,
                                   eval_buffer_state: ReplayBufferState | None = None):
    if sac_kwargs is None:
        sac_kwargs = config.sac_kwargs

    system = LearnedCarSystem(model=bnn_model,
                              include_noise=config.include_aleatoric_noise,
                              predict_difference=config.predict_difference,
                              num_frame_stack=config.num_stacked_actions,
                              **config.car_reward_kwargs)

    if eval_buffer_state is None:
        eval_buffer_state = data_buffer_state

    key, eval_env_key = jax.random.split(key)
    env = BraxWrapper(system=system,
                      sample_buffer_state=data_buffer_state,
                      sample_buffer=data_buffer,
                      system_params=system.init_params(key))

    eval_env = BraxWrapper(system=system,
                           sample_buffer_state=eval_buffer_state,
                           sample_buffer=data_buffer,
                           system_params=system.init_params(eval_env_key))

    # Here we create eval envs
    sac_trainer = SAC(environment=env,
                      eval_environment=eval_env,
                      eval_key_fixed=True,
                      return_best_model=config.return_best_policy,
                      **sac_kwargs, )
    return sac_trainer


def set_up_bnn_dynamics_model(config: Any, key: jax.random.PRNGKey):
    sim = RaceCarSim(encode_angle=True, use_blend=config.sim_prior == 'high_fidelity', car_id=2)
    if config.num_stacked_actions > 0:
        sim = StackedActionSimWrapper(sim, num_stacked_actions=config.num_stacked_actions, action_size=2)
    if config.predict_difference:
        sim = PredictStateChangeWrapper(sim)

    standard_params = {
        'input_size': sim.input_size,
        'output_size': sim.output_size,
        'rng_key': key,
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_data': True,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': bool(config.learnable_likelihood_std),
        'likelihood_exponent': config.likelihood_exponent,
        'hidden_layer_sizes': [64, 64, 64],
        'normalization_stats': sim.normalization_stats,
        'data_batch_size': config.data_batch_size,
        'hidden_activation': jax.nn.leaky_relu,
        'num_train_steps': config.bnn_train_steps,
    }

    if config.sim_prior == 'none_FVSGD':
        bnn = BNN_FSVGD(
            **standard_params,
            domain=sim.domain,
            bandwidth_svgd=config.bandwidth_svgd,
        )
    elif config.sim_prior == 'none_SVGD':
        bnn = BNN_SVGD(
            **standard_params,
            bandwidth_svgd=1.0,
        )
    elif config.sim_prior == 'high_fidelity_no_aditive_GP':
        bnn = BNN_FSVGD_SimPrior(
            **standard_params,
            domain=sim.domain,
            function_sim=sim,
            score_estimator='gp',
            num_f_samples=config.num_f_samples,
            bandwidth_svgd=config.bandwidth_svgd,
            num_measurement_points=config.num_measurement_points,
        )
    else:
        if config.sim_prior == 'high_fidelity':
            outputscales_racecar = [0.008, 0.008, 0.009, 0.009, 0.05, 0.05, 0.20]
        elif config.sim_prior == 'low_fidelity':
            outputscales_racecar = [0.008, 0.008, 0.01, 0.01, 0.08, 0.08, 0.5]
        else:
            raise ValueError(f'Invalid sim prior: {config.sim_prior}')

        sim = AdditiveSim(base_sims=[sim,
                                     GaussianProcessSim(sim.input_size, sim.output_size,
                                                        output_scale=outputscales_racecar,
                                                        length_scale=config.length_scale_aditive_sim_gp,
                                                        consider_only_first_k_dims=None)
                                     ])

        bnn = BNN_FSVGD_SimPrior(
            **standard_params,
            domain=sim.domain,
            function_sim=sim,
            score_estimator='gp',
            num_f_samples=config.num_f_samples,
            bandwidth_svgd=config.bandwidth_svgd,
            num_measurement_points=config.num_measurement_points,
        )
    return bnn


def set_up_dummy_sac_trainer(main_config, mbrl_config: ModelBasedRLConfig, key: jax.random.PRNGKey):
    key, key_buffer_init, key_bnn = jax.random.split(key, 3)

    # data buffer
    true_data_buffer, true_data_buffer_state = init_transition_buffer(config=mbrl_config, key=key_buffer_init)

    # dummy_bnn
    bnn = set_up_bnn_dynamics_model(config=main_config, key=key)

    sac_trainer = set_up_model_based_sac_trainer(
        bnn_model=bnn, data_buffer=true_data_buffer,
        data_buffer_state=true_data_buffer_state, key=key_bnn, config=mbrl_config)

    return sac_trainer


def prepare_init_transitions_for_car_env(key: jax.random.PRNGKey, number_of_samples: int, num_frame_stack: int = 3):
    sim = RCCarSimEnv(encode_angle=True, use_tire_model=True)
    action_dim = 2
    key_init_state = jax.random.split(key, number_of_samples)
    state_obs = jax.vmap(sim.reset)(rng_key=key_init_state)
    framestacked_actions = jnp.zeros(
        shape=(number_of_samples, num_frame_stack * action_dim))
    actions = jnp.zeros(shape=(number_of_samples, action_dim))
    rewards = jnp.zeros(shape=(number_of_samples,))
    discounts = 0.99 * jnp.ones(shape=(number_of_samples,))
    transitions = Transition(observation=jnp.concatenate([state_obs, framestacked_actions], axis=-1),
                             action=actions,
                             reward=rewards,
                             discount=discounts,
                             next_observation=jnp.concatenate([state_obs, framestacked_actions], axis=-1))
    return transitions
