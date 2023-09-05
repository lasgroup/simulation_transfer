import os
import pickle
from typing import Callable, Any

import chex
import cloudpickle
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from jax import jit
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from experiments.data_provider import provide_data_and_sim, _RACECAR_NOISE_STD_ENCODED
from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.models.bnn_svgd import BNN_SVGD
from sim_transfer.rl.model_based_rl.learned_system import LearnedCarSystem
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.util import plot_rc_trajectory


class RLFromOfflineData:
    def __init__(self,
                 data_source: str = 'real_racecar_new_actionstack',
                 data_spec: dict = {'num_samples_train': 10000},
                 bnn_model: BatchedNeuralNetworkModel = None,
                 include_aleatoric_noise: bool = True,
                 predict_difference: bool = True,
                 car_reward_kwargs: dict = None,
                 max_replay_size_true_data_buffer: int = 10 ** 4,
                 sac_kwargs: dict = None,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 return_best_policy: bool = True,
                 ):
        self.return_best_policy = return_best_policy
        self.include_aleatoric_noise = include_aleatoric_noise
        self.data_source = data_source
        self.data_spec = data_spec
        self.bnn_model = bnn_model
        self.predict_difference = predict_difference

        self.car_reward_kwargs = car_reward_kwargs
        self.sac_kwargs = sac_kwargs

        x_train, y_train, x_test, y_test, sim = self.load_data()

        # TODO: rn it is hardcoded
        state_dim = 7
        action_dim = 2
        self.state_dim = state_dim
        self.action_dim = action_dim
        # We compute number of frames to stack
        state_with_frame_stack_dim = x_train.shape[1] - action_dim
        self.state_with_frame_stack_dim = state_with_frame_stack_dim

        self.num_frame_stack = (x_train.shape[1] - state_dim - action_dim) // action_dim

        # Reshape the data and prepare it for training
        states_obs = x_train[:, :state_dim]
        next_state_obs = y_train
        last_actions = x_train[:, state_with_frame_stack_dim:]
        framestacked_actions = x_train[:, state_dim:state_with_frame_stack_dim]
        framestacked_actions = jnp.flip(framestacked_actions, axis=-1)

        # Here we shift frame stacking
        next_framestacked_actions = jnp.roll(framestacked_actions, shift=action_dim, axis=-1)
        next_framestacked_actions = next_framestacked_actions.at[:, :action_dim].set(last_actions)

        rewards = jnp.zeros(shape=(x_train.shape[0],))
        discounts = 0.99 * jnp.ones(shape=(x_train.shape[0],))
        transitions = Transition(observation=jnp.concatenate([states_obs, framestacked_actions], axis=-1),
                                 action=last_actions,
                                 reward=rewards,
                                 discount=discounts,
                                 next_observation=jnp.concatenate([next_state_obs, next_framestacked_actions], axis=-1))

        # We create a dummy sample to init the buffer
        dummy_obs = jnp.zeros(shape=(state_dim + action_dim * self.num_frame_stack,))
        self.dummy_sample = Transition(observation=dummy_obs,
                                       action=jnp.zeros(shape=(action_dim,)),
                                       reward=jnp.array(0.0),
                                       discount=jnp.array(0.99),
                                       next_observation=dummy_obs)

        self.true_data_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)

        # We init and insert the data in the true data buffer
        self.key = key
        self.key, key_init_buffer, key_insert_data = jr.split(self.key, 3)
        true_buffer_state = self.true_data_buffer.init(key_init_buffer)
        true_buffer_state = self.true_data_buffer.insert(true_buffer_state, transitions)
        self.true_buffer_state = true_buffer_state

        # Prepare data to train the model
        self.x_train = self.reshape_xs(x_train)
        self.y_train = y_train
        self.x_test = self.reshape_xs(x_test)
        self.y_test = y_test

        if self.predict_difference:
            self.y_train = self.y_train - self.x_train[..., :7]
            self.y_test = self.y_test - self.x_test[..., :7]

    def reshape_xs(self, xs):
        states_obs = xs[:, :self.state_dim]
        last_actions = xs[:, self.state_with_frame_stack_dim:]
        framestacked_actions = xs[:, self.state_dim:self.state_with_frame_stack_dim]
        framestacked_actions = jnp.flip(framestacked_actions, axis=-1)
        return jnp.concatenate([states_obs, framestacked_actions, last_actions], axis=-1)

    def load_data(self):
        # y_train is the next state not the difference
        x_train, y_train, x_test, y_test, sim = provide_data_and_sim(data_source='real_racecar_new_actionstack',
                                                                     data_spec={'num_samples_train': 10000})
        return x_train, y_train, x_test, y_test, sim

    def train_model(self,
                    learn_std: bool,
                    bnn_train_steps: int,
                    return_best_bnn: bool = True
                    ) -> BNN_SVGD:
        # x_train, y_train, x_test, y_test, sim = self.load_data()
        x_train, y_train, x_test, y_test, sim = self.x_train, self.y_train, self.x_test, self.y_test, None
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # Create a bnn model
        standard_model_params = {
            'input_size': x_train.shape[-1],
            'output_size': y_train.shape[-1],
            'rng_key': jr.PRNGKey(234234345),
            # 'normalization_stats': sim.normalization_stats, TODO: Jonas: adjust sim for normalization stats
            'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
            'normalize_likelihood_std': True,
            'learn_likelihood_std': learn_std,
            'likelihood_exponent': 0.5,
            'hidden_layer_sizes': [64, 64, 64],
            'data_batch_size': 32,
        }

        if self.bnn_model is not None:
            bnn = self.bnn_model
        else:
            bnn = BNN_SVGD(**standard_model_params,
                           bandwidth_svgd=1.0)

        # Train the bnn model
        bnn.fit(x_train=x_train, y_train=y_train, x_eval=x_test, y_eval=y_test, log_to_wandb=True,
                keep_the_best=return_best_bnn, metrics_objective='eval_nll', num_steps=bnn_train_steps)

        return bnn

    def train_policy(self,
                     bnn_model: BatchedNeuralNetworkModel,
                     true_data_buffer_state: ReplayBufferState,
                     key: chex.PRNGKey):

        system = LearnedCarSystem(model=bnn_model,
                                  include_noise=self.include_aleatoric_noise,
                                  predict_difference=self.predict_difference,
                                  num_frame_stack=self.num_frame_stack,
                                  **self.car_reward_kwargs)

        key_train, key_simulate, *keys_sys_params = jr.split(key, 4)
        env = BraxWrapper(system=system,
                          sample_buffer_state=true_data_buffer_state,
                          sample_buffer=self.true_data_buffer,
                          system_params=system.init_params(keys_sys_params[0]))

        _sac_kwargs = self.sac_kwargs
        sac_trainer = SAC(environment=env,
                          eval_environment=env,
                          eval_key_fixed=True,
                          return_best_model=self.return_best_policy,
                          **_sac_kwargs, )

        params, metrics = sac_trainer.run_training(key=key_train)

        make_inference_fn = sac_trainer.make_policy

        @jit
        def policy(x):
            return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]

        return policy, params, metrics

    def prepare_policy(self, params: Any | None = None, filename: str = None):
        if params is None:
            with open(filename, 'rb') as handle:
                params = pickle.load(handle)

        x_train, y_train, x_test, y_test, sim = self.x_train, self.y_train, self.x_test, self.y_test, None
        # Create a bnn model
        standard_model_params = {
            'input_size': x_train.shape[-1],
            'output_size': y_train.shape[-1],
            'rng_key': jr.PRNGKey(234234345),
            # 'normalization_stats': sim.normalization_stats, TODO: Jonas: adjust sim for normalization stats
            'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
            'normalize_likelihood_std': True,
            'learn_likelihood_std': True,
            'likelihood_exponent': 0.5,
            'hidden_layer_sizes': [64, 64, 64],
            'data_batch_size': 32,
        }
        bnn = BNN_SVGD(**standard_model_params,
                       bandwidth_svgd=1.0)
        system = LearnedCarSystem(model=bnn,
                                  include_noise=self.include_aleatoric_noise,
                                  predict_difference=self.predict_difference,
                                  num_frame_stack=self.num_frame_stack,
                                  **self.car_reward_kwargs)

        key_train, key_simulate, *keys_sys_params = jr.split(self.key, 4)
        env = BraxWrapper(system=system,
                          sample_buffer_state=self.true_buffer_state,
                          sample_buffer=self.true_data_buffer,
                          system_params=system.init_params(keys_sys_params[0]))

        _sac_kwargs = self.sac_kwargs
        sac_trainer = SAC(environment=env,
                          eval_environment=env,
                          eval_key_fixed=True,
                          return_best_model=self.return_best_policy,
                          **_sac_kwargs, )
        make_inference_fn = sac_trainer.make_policy

        @jit
        def policy(x):
            return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]

        return policy

    def prepare_policy_from_offline_data(self,
                                         learn_std: bool = True,
                                         bnn_train_steps: int = 10_000,
                                         return_best_bnn: bool = True):
        bnn_model = self.train_model(learn_std=learn_std, bnn_train_steps=bnn_train_steps,
                                     return_best_bnn=return_best_bnn)
        policy, params, metrics = self.train_policy(bnn_model, self.true_buffer_state, self.key)

        # Save policy parameters
        directory = os.path.join(wandb.run.dir, 'models')
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join('models', 'parameters.pkl')
        with open(os.path.join(wandb.run.dir, model_path), 'wb') as handle:
            pickle.dump(params, handle)
        wandb.save(os.path.join(wandb.run.dir, model_path), wandb.run.dir)

        directory = os.path.join(wandb.run.dir, 'models')
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join('models', 'bnn_model.pkl')
        with open(os.path.join(wandb.run.dir, model_path), 'wb') as handle:
            pickle.dump(bnn_model, handle)
        wandb.save(os.path.join(wandb.run.dir, model_path), wandb.run.dir)

        return policy, params, metrics, bnn_model

    def evaluate_policy(self,
                        policy: Callable,
                        bnn_model: BatchedNeuralNetworkModel,
                        key=jr.PRNGKey(0)):
        sim = RCCarSimEnv(encode_angle=True, use_tire_model=True)
        eval_horizon = 200
        # Now we simulate the policy on the learned model

        obs = sim.reset()
        # Obs represents samples from the init_states_buffer which is of dim x_dim + u_dim * num_frame_stack
        stacked_actions = jnp.zeros(shape=(self.num_frame_stack * self.action_dim,))

        transitions = []
        for step in range(eval_horizon):
            key, subkey = jr.split(key)
            action = policy(jnp.concatenate([obs, stacked_actions], axis=-1))
            z = jnp.concatenate([obs, stacked_actions, action], axis=-1)
            z = z.reshape((1, -1))
            delta_x_dist = bnn_model.predict_dist(z, include_noise=True)
            delta_x = delta_x_dist.sample(seed=subkey)
            delta_x = delta_x.reshape(-1)
            next_obs = obs + delta_x

            transitions.append(Transition(observation=obs,
                                          action=action,
                                          reward=jnp.array(0.0),
                                          discount=jnp.array(0.99),
                                          next_observation=next_obs))
            # We prepare set of new inputs
            obs = next_obs
            # Now we shift the actions
            stacked_actions = jnp.roll(stacked_actions, shift=self.action_dim)
            stacked_actions = stacked_actions.at[:self.action_dim].set(action)

        concatenated_transitions = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *transitions)
        rewards = jnp.sum(concatenated_transitions.reward)
        fig, axes = plot_rc_trajectory(concatenated_transitions.next_observation,
                                       concatenated_transitions.action, encode_angle=True,
                                       show=False)
        wandb.log({'Trajectory_on_learned_model': wandb.Image(fig),
                   'reward_on_learned_model': rewards})
        plt.close('all')


if __name__ == '__main__':
    wandb.init(
        project="Race car test MBRL",
        group='test',
    )

    car_reward_kwargs = dict(encode_angle=True,
                             ctrl_cost_weight=0.005,
                             margin_factor=20)

    NUM_ENV_STEPS_BETWEEN_UPDATES = 16
    NUM_ENVS = 64
    sac_num_env_steps = 20_000
    horizon_len = 50

    SAC_KWARGS = dict(num_timesteps=sac_num_env_steps,
                      num_evals=20,
                      reward_scaling=10,
                      episode_length=horizon_len,
                      episode_length_eval=2 * horizon_len,
                      action_repeat=1,
                      discounting=0.99,
                      lr_policy=3e-4,
                      lr_alpha=3e-4,
                      lr_q=3e-4,
                      num_envs=NUM_ENVS,
                      batch_size=64,
                      grad_updates_per_step=NUM_ENV_STEPS_BETWEEN_UPDATES * NUM_ENVS,
                      num_env_steps_between_updates=NUM_ENV_STEPS_BETWEEN_UPDATES,
                      tau=0.005,
                      wd_policy=0,
                      wd_q=0,
                      wd_alpha=0,
                      num_eval_envs=2 * NUM_ENVS,
                      max_replay_size=5 * 10 ** 4,
                      min_replay_size=2 ** 11,
                      policy_hidden_layer_sizes=(64, 64),
                      critic_hidden_layer_sizes=(64, 64),
                      normalize_observations=True,
                      deterministic_eval=True,
                      wandb_logging=True)

    rl_from_offline_data = RLFromOfflineData(
        sac_kwargs=SAC_KWARGS,
        car_reward_kwargs=car_reward_kwargs)
    policy, params, metrics, bnn_model = rl_from_offline_data.prepare_policy_from_offline_data(learn_std=True,
                                                                                               bnn_train_steps=2_000)
    filename_params = os.path.join(wandb.run.dir, 'models/policy.pkl')
    filename_bnn_model = os.path.join(wandb.run.dir, 'models/bnn_model.pkl')

    with open(filename_bnn_model, 'rb') as handle:
        bnn_model = cloudpickle.load(handle)

    rl_from_offline_data.evaluate_policy(policy, bnn_model, key=jr.PRNGKey(0))
