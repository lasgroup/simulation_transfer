import os
import pickle
from typing import Callable, Any

import chex
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from jax import jit, vmap
from jax.lax import scan
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
                 x_train: chex.Array,
                 y_train: chex.Array,
                 x_test: chex.Array,
                 y_test: chex.Array,
                 bnn_model: BatchedNeuralNetworkModel = None,
                 include_aleatoric_noise: bool = True,
                 predict_difference: bool = True,
                 car_reward_kwargs: dict = None,
                 max_replay_size_true_data_buffer: int = 30 ** 4,
                 sac_kwargs: dict = None,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 return_best_policy: bool = True,
                 num_frame_stack: int = 3,
                 test_data_ratio: float = 0.1,
                 num_init_points_to_bs_for_sac_learning: int | None = 100,
                 eval_sac_only_from_init_states: bool = False,
                 eval_bnn_model_on_all_offline_data: bool = True,
                 train_sac_only_from_init_states: bool = False,
                 load_pretrained_bnn_model: bool = True,
                 ):
        self.eval_bnn_model_on_all_offline_data = eval_bnn_model_on_all_offline_data
        self.eval_sac_only_from_init_states = eval_sac_only_from_init_states
        self.train_sac_only_from_init_states = train_sac_only_from_init_states
        self.load_pretrained_bnn_model = load_pretrained_bnn_model

        # We load the model trained on dataset of 20_000 points for evaluation
        if self.load_pretrained_bnn_model:
            simulation_transfer_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            bnn_dir = os.path.join(simulation_transfer_dir, 'bnn_models_pretrained')
            bnn_model_path = os.path.join(bnn_dir, 'bnn_svgd_model_on_20_000_points.pkl')

            with open(bnn_model_path, 'rb') as handle:
                bnn_model_pretrained = pickle.load(handle)

            self.bnn_model_pretrained = bnn_model_pretrained
        else:
            self.bnn_model_pretrained = None
        self.test_data_ratio = test_data_ratio
        self.key = key

        self.return_best_policy = return_best_policy
        self.include_aleatoric_noise = include_aleatoric_noise
        self.bnn_model = bnn_model
        self.predict_difference = predict_difference

        self.car_reward_kwargs = car_reward_kwargs
        self.sac_kwargs = sac_kwargs

        # We split the train data into train and eval
        self.key, key_split = jr.split(self.key)
        x_train, y_train, x_eval, y_eval = self.shuffle_and_split_data(x_train,
                                                                       y_train,
                                                                       self.test_data_ratio,
                                                                       key_split)

        # Prepare number of init points for learning
        if num_init_points_to_bs_for_sac_learning is None:
            num_init_points_to_bs_for_sac_learning = x_train.shape[0]
        self.num_init_points_to_bs_for_learning = num_init_points_to_bs_for_sac_learning

        state_dim = 7
        action_dim = 2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_frame_stack = num_frame_stack
        # We compute number of frames to stack
        self.state_with_frame_stack_dim = self.num_frame_stack * action_dim + state_dim

        # Reshape the data and prepare it for training
        states_obs = x_train[:, :state_dim]
        next_state_obs = y_train
        last_actions = x_train[:, self.state_with_frame_stack_dim:]
        framestacked_actions = x_train[:, state_dim:self.state_with_frame_stack_dim]
        framestacked_actions = self.revert_order_of_stacked_actions(framestacked_actions)

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
        self.key, key_init_buffer, key_insert_data = jr.split(self.key, 3)
        true_buffer_state = self.true_data_buffer.init(key_init_buffer)
        true_buffer_state = self.true_data_buffer.insert(true_buffer_state, transitions)
        self.true_buffer_state = true_buffer_state

        # Prepare data to train the model
        self.x_train = self.reshape_xs(x_train)
        self.y_train = y_train
        self.x_test = self.reshape_xs(x_test)
        self.y_test = y_test
        self.x_eval = self.reshape_xs(x_eval)
        self.y_eval = y_eval

        if self.predict_difference:
            self.y_train = self.y_train - self.x_train[..., :self.state_dim]
            self.y_eval = self.y_eval - self.x_eval[..., :self.state_dim]
            self.y_test = self.y_test - self.x_test[..., :self.state_dim]

    def reshape_xs(self, xs):
        states_obs = xs[:, :self.state_dim]
        last_actions = xs[:, self.state_with_frame_stack_dim:]
        framestacked_actions = xs[:, self.state_dim:self.state_with_frame_stack_dim]
        framestacked_actions = self.revert_order_of_stacked_actions(framestacked_actions)
        return jnp.concatenate([states_obs, framestacked_actions, last_actions], axis=-1)

    @staticmethod
    def shuffle_and_split_data(x_data, y_data, test_ratio, key: chex.PRNGKey):
        # Get the size of the data
        num_data = x_data.shape[0]

        # Create a permutation of indices
        perm = jr.permutation(key, num_data)
        # Permute the data
        x_data = x_data[perm]
        y_data = y_data[perm]

        # Calculate the number of examples in the test set
        num_test = int(test_ratio * num_data)

        # Calculate the number of examples in the train set
        num_train = num_data - num_test

        # Split the data
        x_train = x_data[:num_train]
        x_test = x_data[num_train:]
        y_train = y_data[:num_train]
        y_test = y_data[num_train:]

        return x_train, y_train, x_test, y_test

    def train_model(self,
                    bnn_train_steps: int,
                    return_best_bnn: bool = True
                    ) -> BNN_SVGD:
        x_train, y_train, x_eval, y_eval, sim = self.x_train, self.y_train, self.x_eval, self.y_eval, None
        x_test, y_test = self.x_test, self.y_test
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        bnn = self.bnn_model
        if self.test_data_ratio == 0.0:
            metrics_objective = 'train_nll_loss'
            x_eval, y_eval = x_test, y_test
        else:
            metrics_objective = 'eval_nll'

        # Train the bnn model
        bnn.fit(x_train=x_train, y_train=y_train, x_eval=x_eval, y_eval=y_eval, log_to_wandb=True,
                keep_the_best=return_best_bnn, metrics_objective=metrics_objective, num_steps=bnn_train_steps)
        return bnn

    def prepare_init_transitions(self, key: chex.PRNGKey, number_of_samples: int):
        sim = RCCarSimEnv(encode_angle=True, use_tire_model=True)

        key_init_state = jr.split(key, number_of_samples)
        state_obs = vmap(sim.reset)(rng_key=key_init_state)
        framestacked_actions = jnp.zeros(
            shape=(number_of_samples, self.num_frame_stack * self.action_dim))
        actions = jnp.zeros(shape=(number_of_samples, self.action_dim))
        rewards = jnp.zeros(shape=(number_of_samples,))
        discounts = 0.99 * jnp.ones(shape=(number_of_samples,))
        transitions = Transition(observation=jnp.concatenate([state_obs, framestacked_actions], axis=-1),
                                 action=actions,
                                 reward=rewards,
                                 discount=discounts,
                                 next_observation=jnp.concatenate([state_obs, framestacked_actions], axis=-1))
        return transitions

    def train_policy(self,
                     bnn_model: BatchedNeuralNetworkModel,
                     true_data_buffer_state: ReplayBufferState,
                     key: chex.PRNGKey):

        system = LearnedCarSystem(model=bnn_model,
                                  include_noise=self.include_aleatoric_noise,
                                  predict_difference=self.predict_difference,
                                  num_frame_stack=self.num_frame_stack,
                                  **self.car_reward_kwargs)

        key_train, key_simulate, key_init_state, *keys_sys_params = jr.split(key, 5)

        # Here we add init points to the true_data_buffer
        key_init_state, key_init_buffer = jr.split(key_init_state)
        init_transitions = self.prepare_init_transitions(key_init_state, self.num_init_points_to_bs_for_learning)

        if self.train_sac_only_from_init_states:
            train_buffer_state = self.true_data_buffer.init(key_init_buffer)
            train_buffer_state = self.true_data_buffer.insert(train_buffer_state, init_transitions)
        else:
            train_buffer_state = self.true_data_buffer.insert(true_data_buffer_state, init_transitions)

        env = BraxWrapper(system=system,
                          sample_buffer_state=train_buffer_state,
                          sample_buffer=self.true_data_buffer,
                          system_params=system.init_params(keys_sys_params[0]))
        init_states_bs = train_buffer_state
        if self.eval_sac_only_from_init_states:
            init_states_bs = self.true_data_buffer.init(key_init_buffer)
            init_states_bs = self.true_data_buffer.insert(init_states_bs, init_transitions)

        eval_env = BraxWrapper(system=system,
                               sample_buffer_state=init_states_bs,
                               sample_buffer=self.true_data_buffer,
                               system_params=system.init_params(keys_sys_params[0]))

        _sac_kwargs = self.sac_kwargs
        sac_trainer = SAC(environment=env,
                          eval_environment=eval_env,
                          eval_key_fixed=True,
                          return_best_model=self.return_best_policy,
                          **_sac_kwargs, )

        params, metrics = sac_trainer.run_training(key=key_train)

        make_inference_fn = sac_trainer.make_policy

        @jit
        def policy(x):
            return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]

        return policy, params, metrics

    def revert_order_of_stacked_actions(self, stacked_actions: chex.Array):
        assert self.action_dim * self.num_frame_stack == stacked_actions.shape[-1]
        if self.num_frame_stack == 0:
            return stacked_actions
        actions_list = []
        # We split the actions into a list of actions
        for i in range(self.num_frame_stack):
            actions_list.append(stacked_actions[..., 2 * i:2 * i + 2])
        # We revert list now
        actions_list = actions_list[::-1]
        return jnp.concatenate(actions_list, axis=-1)

    def prepare_policy(self, params: Any | None = None, filename: str = None):
        if params is None:
            with open(filename, 'rb') as handle:
                params = pickle.load(handle)

        x_train, y_train, x_test, y_test, sim = self.x_train, self.y_train, self.x_eval, self.y_eval, None
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

    def evaluate_bnn_model_on_all_collected_data(self, bnn_model: BatchedNeuralNetworkModel):
        data_source: str = 'real_racecar_new_actionstack'
        data_spec: dict = {'num_samples_train': 20000}
        x_data, y_data, _, _, sim = provide_data_and_sim(data_source=data_source,
                                                         data_spec=data_spec)
        x_data = self.reshape_xs(x_data)
        if self.predict_difference:
            y_data = y_data - x_data[..., :self.state_dim]
        eval_stats = bnn_model.eval(x_data, y_data, per_dim_metrics=True, prefix='eval_on_all_offline_data/')
        wandb.log(eval_stats)

    def eval_bnn_model_on_test_data(self, bnn_model: BatchedNeuralNetworkModel):
        x_test, y_test = self.x_test, self.y_test
        test_stats = bnn_model.eval(x_test, y_test, per_dim_metrics=True, prefix='test_data/')
        wandb.log(test_stats)

    def prepare_policy_from_offline_data(self,
                                         bnn_train_steps: int = 10_000,
                                         return_best_bnn: bool = True):
        bnn_model = self.train_model(bnn_train_steps=bnn_train_steps,
                                     return_best_bnn=return_best_bnn)
        if self.eval_bnn_model_on_all_offline_data:
            self.evaluate_bnn_model_on_all_collected_data(bnn_model)
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

    @staticmethod
    def arg_mean(a: chex.Array):
        # Return index of the element that is closest to the mean
        return jnp.argmin(jnp.abs(a - jnp.mean(a)))

    def evaluate_policy_on_the_simulator(self,
                                         policy: Callable,
                                         key: chex.PRNGKey = jr.PRNGKey(0),
                                         num_evals: int = 1, ):
        def reward_on_simulator(key: chex.PRNGKey):
            actions_buffer = jnp.zeros(shape=(self.action_dim * self.num_frame_stack))
            sim = RCCarSimEnv(encode_angle=True, use_tire_model=True,
                              margin_factor=self.car_reward_kwargs['margin_factor'],
                              ctrl_cost_weight=self.car_reward_kwargs['ctrl_cost_weight'], )
            obs = sim.reset(key)
            done = False
            transitions_for_plotting = []
            while not done:
                policy_input = jnp.concatenate([obs, actions_buffer], axis=-1)
                action = policy(policy_input)
                next_obs, reward, done, info = sim.step(action)
                # Prepare new actions buffer
                # TODO: This is OLD version of action stacking!
                if self.num_frame_stack > 0:
                    next_actions_buffer = jnp.concatenate([action, actions_buffer[:-self.action_dim]])
                else:
                    next_actions_buffer = jnp.zeros(shape=(0,))

                # if self.num_frame_stack > 0:
                #     next_actions_buffer = jnp.concatenate([actions_buffer[self.action_dim:], action])
                # else:
                #     next_actions_buffer = jnp.zeros(shape=(0,))

                transitions_for_plotting.append(Transition(observation=obs,
                                                           action=action,
                                                           reward=jnp.array(reward),
                                                           discount=jnp.array(0.99),
                                                           next_observation=next_obs)
                                                )
                actions_buffer = next_actions_buffer
                obs = next_obs

            concatenated_transitions_for_plotting = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0),
                                                                 *transitions_for_plotting)
            reward_on_simulator = jnp.sum(concatenated_transitions_for_plotting.reward)
            return reward_on_simulator, concatenated_transitions_for_plotting

        rewards, trajectories = vmap(reward_on_simulator)(jr.split(key, num_evals))

        reward_mean = jnp.mean(rewards)
        reward_std = jnp.std(rewards)

        reward_mean_index = self.arg_mean(rewards)

        transitions_mean = jtu.tree_map(lambda x: x[reward_mean_index], trajectories)
        fig, axes = plot_rc_trajectory(transitions_mean.next_observation,
                                       transitions_mean.action, encode_angle=True,
                                       show=False)
        model_name = 'simulator'
        wandb.log({f'Mean_trajectory_on_{model_name}': wandb.Image(fig),
                   f'reward_mean_on_{model_name}': float(reward_mean),
                   f'reward_std_on_{model_name}': float(reward_std)})
        plt.close('all')

    def evaluate_policy(self,
                        policy: Callable,
                        bnn_model: BatchedNeuralNetworkModel | None = None,
                        key: chex.PRNGKey = jr.PRNGKey(0),
                        num_evals: int = 1, ):
        init_stacked_actions = jnp.zeros(shape=(self.num_frame_stack * self.action_dim,))
        model_name = 'pretrained_model' if bnn_model is None else 'learned_model'

        sim = RCCarSimEnv(encode_angle=True, use_tire_model=True)
        eval_horizon = 200
        # Now we simulate the policy on the learned model

        key_init_obs, key_generate_trajectories = jr.split(key)
        key_init_obs = jr.split(key_init_obs, num_evals)
        obs = vmap(sim.reset)(rng_key=key_init_obs)
        if bnn_model is None:
            bnn_model = self.bnn_model_pretrained
            if bnn_model is None:
                raise ValueError('You have not loaded the pretrained model.')
        learned_car_system = LearnedCarSystem(model=bnn_model,
                                              include_noise=self.include_aleatoric_noise,
                                              predict_difference=self.predict_difference,
                                              num_frame_stack=self.num_frame_stack,
                                              **self.car_reward_kwargs)

        def f_step(carry, _):
            state, sys_params = carry
            action = policy(state)
            sys_state = learned_car_system.step(x=state, u=action, system_params=sys_params)
            new_state = sys_state.x_next
            transition = Transition(observation=state[:self.state_dim],
                                    action=action,
                                    reward=sys_state.reward,
                                    discount=jnp.array(0.99),
                                    next_observation=new_state[:self.state_dim], )
            new_carry = (new_state, sys_state.system_params)
            return new_carry, transition

        key_generate_trajectories = jr.split(key_generate_trajectories, num_evals)

        def get_trajectory_transitions(init_obs, key):
            sys_params = learned_car_system.init_params(key)
            state = jnp.concatenate([init_obs, init_stacked_actions], axis=-1)
            last_carry, transitions = scan(f_step, (state, sys_params), None, length=eval_horizon)
            return transitions

        trajectories = vmap(get_trajectory_transitions)(obs, key_generate_trajectories)

        # Now we calculate mean reward and std of rewards
        rewards = jnp.sum(trajectories.reward, axis=-1)
        reward_mean = jnp.mean(rewards)
        reward_std = jnp.std(rewards)

        reward_mean_index = self.arg_mean(rewards)

        transitions_mean = jtu.tree_map(lambda x: x[reward_mean_index], trajectories)
        fig, axes = plot_rc_trajectory(transitions_mean.next_observation,
                                       transitions_mean.action, encode_angle=True,
                                       show=False)

        wandb.log({f'Mean_trajectory_on_{model_name}': wandb.Image(fig),
                   f'reward_mean_on_{model_name}': float(reward_mean),
                   f'reward_std_on_{model_name}': float(reward_std)})
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

    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
        data_source='racecar_actionstack',
        data_spec={'num_samples_train': 10_000,
                   'use_hf_sim': True, },
        data_seed=1234
    )

    standard_params = {
        'input_size': sim.input_size,
        'output_size': sim.output_size,
        'rng_key': jr.PRNGKey(0),
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_data': True,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': True,
        'likelihood_exponent': 0.5,
        'hidden_layer_sizes': [64, 64, 64],
        'data_batch_size': 32,
    }

    model = BNN_SVGD(
        **standard_params,
        num_train_steps=2_000,
    )

    rl_from_offline_data = RLFromOfflineData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        sac_kwargs=SAC_KWARGS,
        car_reward_kwargs=car_reward_kwargs,
        bnn_model=model,
        test_data_ratio=0.1,
        num_init_points_to_bs_for_sac_learning=100,
        eval_sac_only_from_init_states=True,
    )
    policy, params, metrics, bnn_model = rl_from_offline_data.prepare_policy_from_offline_data(bnn_train_steps=2_000)
    rl_from_offline_data.evaluate_policy_on_the_simulator(policy, key=jr.PRNGKey(0), num_evals=100)
    # filename_params = os.path.join(wandb.run.dir, 'models/policy.pkl')
    # filename_bnn_model = os.path.join(wandb.run.dir, 'models/bnn_svgd_model_on_20_000_points.pkl')
    #
    # with open(filename_bnn_model, 'rb') as handle:
    #     bnn_model = pickle.load(handle)

    # rl_from_offline_data.evaluate_policy(policy, key=jr.PRNGKey(0), num_evals=100)
    # rl_from_offline_data.evaluate_policy(policy, bnn_model=bnn_model, key=jr.PRNGKey(0), num_evals=100)

    wandb.finish()
