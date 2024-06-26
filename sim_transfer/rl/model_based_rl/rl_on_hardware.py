import os
import pickle
from typing import Callable
from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import wandb
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from jaxtyping import PyTree
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.hardware.car_env import CarEnv
from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.models.bnn_svgd import BNN_SVGD
from sim_transfer.rl.model_based_rl.learned_system import LearnedCarSystem
from sim_transfer.rl.model_based_rl.utils import split_data
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.util import plot_rc_trajectory


class RealCarRL:
    def __init__(self,
                 gym_env: CarEnv,
                 bnn_model: BatchedNeuralNetworkModel = None,
                 offline_data: Transition = None,
                 max_replay_size_true_data_buffer: int = 10 ** 4,
                 include_aleatoric_noise: bool = True,
                 car_reward_kwargs: dict = None,
                 sac_kwargs: dict = None,
                 discounting: chex.Array = jnp.array(0.99),
                 reset_bnn: bool = True,
                 return_best_bnn: bool = True,
                 return_best_policy: bool = True,
                 predict_difference: bool = True,
                 bnn_training_test_ratio: float = 0.2,
                 num_frame_stack: int = 3,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 ):
        # Input dimension of bnn_model is u_dim + x_dim * num_frame_stack
        self.num_frame_stack = num_frame_stack
        self.bnn_training_test_ratio = bnn_training_test_ratio
        self.predict_difference = predict_difference
        self.return_best_policy = return_best_policy
        self.return_best_bnn = return_best_bnn
        self.reset_bnn = reset_bnn
        self.discounting = discounting
        self.car_reward_kwargs = car_reward_kwargs
        self.sac_kwargs = sac_kwargs
        self.include_aleatoric_noise = include_aleatoric_noise
        self.gym_env = gym_env
        self.bnn_model = bnn_model
        self.key = key

        self.state_dim = 7
        self.action_dim = 2

        dummy_obs = jnp.zeros(shape=(self.state_dim + self.action_dim * self.num_frame_stack,))
        self.dummy_sample = Transition(observation=dummy_obs,
                                       action=jnp.zeros(shape=(self.action_dim,)),
                                       reward=jnp.array(0.0),
                                       discount=jnp.array(0.99),
                                       next_observation=dummy_obs)

        self.true_data_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)

        # We now insert data into the true data buffer
        self.key, key_bf_init = jr.split(self.key)
        true_data_bf_state = self.true_data_buffer.init(key_bf_init)

        assert offline_data.observation.shape[-1] == self.state_dim + self.action_dim * self.num_frame_stack
        self.true_data_buffer_state = self.add_data_to_buffer(data=offline_data,
                                                              true_data_buffer_state=true_data_bf_state)

        self.init_states_buffer = UniformSamplingQueue(
            max_replay_size=100,  # Should be larger than the number of episodes we run
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)

    def add_data_to_buffer(self, data: Transition, true_data_buffer_state: ReplayBufferState) -> ReplayBufferState:
        return self.true_data_buffer.insert(true_data_buffer_state, data)

    def define_wandb_metrics(self):
        wandb.define_metric('x_axis/episode')

    def train_policy(self,
                     bnn_model: BatchedNeuralNetworkModel,
                     true_data_buffer_state: ReplayBufferState,
                     init_states_buffer_state: ReplayBufferState,
                     key: chex.PRNGKey,
                     episode_idx: int) -> Tuple[Callable[[chex.Array], chex.Array], PyTree]:
        _sac_kwargs = self.sac_kwargs
        # if episode_idx == 0:
        #     _sac_kwargs = copy.deepcopy(_sac_kwargs)
        #     _sac_kwargs['num_timesteps'] = 10_000
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

        # Here we create eval envs
        sac_trainer = SAC(environment=env,
                          eval_environment=env,
                          eval_key_fixed=True,
                          return_best_model=self.return_best_policy,
                          **_sac_kwargs, )

        params, metrics = sac_trainer.run_training(key=key_train)

        best_reward = np.max([summary['eval/episode_reward'] for summary in metrics])
        wandb.log({'best_trained_reward': best_reward,
                   'x_axis/episode': episode_idx})

        make_inference_fn = sac_trainer.make_policy

        def policy(x):
            return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]

        if episode_idx > 0:
            eval_horizon = self.gym_env.max_steps
            # Now we simulate the policy on the learned model
            new_init_state_bs, init_trans = self.init_states_buffer.sample(init_states_buffer_state)
            init_trans = jtu.tree_map(lambda x: x[0], init_trans)
            obs = init_trans.observation
            # Obs represents samples from the init_states_buffer which is of dim x_dim + u_dim * num_frame_stack

            sys_params = system.init_params(keys_sys_params[1])
            transitions = []

            for step in range(eval_horizon):
                action = policy(obs)
                next_sys_state = system.step(x=obs, u=action, system_params=sys_params)
                transitions.append(Transition(observation=obs[:self.state_dim],
                                              action=action,
                                              reward=next_sys_state.reward,
                                              discount=self.discounting,
                                              next_observation=next_sys_state.x_next[:self.state_dim]))
                # We prepare set of new inputs
                obs = next_sys_state.x_next
                sys_params = next_sys_state.system_params

            concatenated_transitions = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *transitions)
            rewards = jnp.sum(concatenated_transitions.reward)
            fig, axes = plot_rc_trajectory(concatenated_transitions.next_observation,
                                           concatenated_transitions.action, encode_angle=self.gym_env.encode_angle,
                                           show=False)
            wandb.log({'Trajectory_on_learned_model': wandb.Image(fig),
                       'reward_on_learned_model': rewards,
                       'x_axis/episode': episode_idx})
            plt.close('all')

        return policy, params

    def simulate_on_true_envs(self,
                              episode_idx: int,
                              policy: Callable[[chex.Array], chex.Array],
                              key: chex.PRNGKey) -> Transition:
        transitions = []
        transitions_for_plotting = []
        actions_buffer = jnp.zeros(shape=(self.action_dim * self.num_frame_stack))
        obs = self.gym_env.reset()
        done = False
        while not done:
            policy_input = jnp.concatenate([obs, actions_buffer], axis=-1)
            action = policy(policy_input)
            next_obs, reward, done, info = self.gym_env.step(action)
            assert next_obs.shape == (self.state_dim,)
            transitions_for_plotting.append(Transition(observation=obs,
                                                       action=action,
                                                       reward=jnp.array(reward),
                                                       discount=self.discounting,
                                                       next_observation=next_obs))
            # Prepare new actions buffer
            next_actions_buffer = jnp.roll(actions_buffer, shift=self.action_dim)
            next_actions_buffer = next_actions_buffer.at[:self.action_dim].set(action)

            transitions.append(Transition(observation=jnp.concatenate([obs, actions_buffer], axis=-1),
                                          action=action,
                                          reward=jnp.array(reward),
                                          discount=self.discounting,
                                          next_observation=jnp.concatenate([next_obs, next_actions_buffer], axis=-1), ))
            actions_buffer = next_actions_buffer
            obs = next_obs

        concatenated_transitions_for_plotting = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0),
                                                             *transitions_for_plotting)
        reward_from_trajectory = jnp.sum(concatenated_transitions_for_plotting.reward)
        # We add now the
        # reward_terminal = info['terminal_reward']
        # Todo: add reward terminal in the simulation as well
        reward_terminal = 0
        reward_on_true_system = reward_from_trajectory + reward_terminal
        print('Reward on true system:', reward_on_true_system)
        fig, axes = plot_rc_trajectory(concatenated_transitions_for_plotting.next_observation,
                                       concatenated_transitions_for_plotting.action,
                                       encode_angle=self.gym_env.encode_angle,
                                       show=False)
        wandb.log({'True_trajectory_path': wandb.Image(fig),
                   'reward_on_true_system': reward_on_true_system,
                   'x_axis/episode': episode_idx})
        plt.close('all')

        concatenated_transitions_for_buffer = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0),
                                                           *transitions)

        return concatenated_transitions_for_buffer

    def train_transition_model(self,
                               true_buffer_state: ReplayBufferState,
                               key: chex.PRNGKey, ) -> BatchedNeuralNetworkModel:
        # Prepare data
        buffer_size = self.true_data_buffer.size(true_buffer_state)
        all_data = true_buffer_state.data[:buffer_size]
        all_transitions = self.true_data_buffer._unflatten_fn(all_data)

        state_action_buffer_pairs = all_transitions.observation
        all_actions = all_transitions.action

        next_state_action_buffer_pairs = all_transitions.next_observation
        x_all = jnp.concatenate([state_action_buffer_pairs, all_actions], axis=-1)
        if self.predict_difference:
            y_all = next_state_action_buffer_pairs[:, :self.state_dim] - state_action_buffer_pairs[:, :self.state_dim]
        else:
            y_all = next_state_action_buffer_pairs[:, :self.state_dim]
        key_split_data, key_reinit_model = jr.split(key)
        x_train, x_test, y_train, y_test = split_data(x_all, y_all,
                                                      test_ratio=self.bnn_training_test_ratio,
                                                      key=key_split_data)

        # Train model
        if self.reset_bnn:
            self.bnn_model.reinit(rng_key=key_reinit_model)
        self.bnn_model.fit(x_train=x_train, y_train=y_train, x_eval=x_test, y_eval=y_test, log_to_wandb=True,
                           keep_the_best=self.return_best_bnn, metrics_objective='eval_nll')
        return self.bnn_model

    def do_episode(self,
                   episode_idx: int,
                   bnn_model: BatchedNeuralNetworkModel,
                   true_buffer_state: ReplayBufferState,
                   init_states_buffer_state: ReplayBufferState,
                   key: chex.PRNGKey):
        # We open the folder

        # Train transition model on the true data buffer only if episode_idx > 0
        if episode_idx == 0:
            print(f"Training bnn model in the episode {episode_idx}")
            bnn_model = self.train_transition_model(true_buffer_state=true_buffer_state, key=key)

        # Train policy on current model
        print("Training policy")
        policy, policy_parameters = self.train_policy(bnn_model=bnn_model,
                                                      true_data_buffer_state=true_buffer_state,
                                                      init_states_buffer_state=init_states_buffer_state,
                                                      key=key,
                                                      episode_idx=episode_idx)
        # Here we store the bnn_model, true_data_buffer, init_states_buffer
        self.save_current_state(policy_parameters=policy_parameters,
                                bnn_model=bnn_model,
                                true_data_buffer_state=true_buffer_state,
                                init_states_buffer_state=init_states_buffer_state,
                                episode_idx=episode_idx)

        # Simulate on true envs with policy and collect data
        print("Simulating on true envs")
        transitions = self.simulate_on_true_envs(episode_idx=episode_idx, policy=policy, key=key)
        # Update true data buffer
        print("Updating true data buffer")
        new_true_buffer_state = self.true_data_buffer.insert(true_buffer_state, transitions)
        print("Updating init states buffer")
        init_states_buffer_state = self.init_states_buffer.insert(init_states_buffer_state,
                                                                  jtu.tree_map(lambda x: x[:1], transitions))
        # Update model
        print("Training bnn model")
        new_bnn_model = self.train_transition_model(true_buffer_state=new_true_buffer_state, key=key)
        return new_bnn_model, new_true_buffer_state, init_states_buffer_state

    @staticmethod
    def save_current_state(policy_parameters: PyTree,
                           bnn_model: BatchedNeuralNetworkModel,
                           true_data_buffer_state: ReplayBufferState,
                           init_states_buffer_state: ReplayBufferState,
                           episode_idx: int):
        # Prepare directory
        directory = os.path.join(wandb.run.dir, 'models_episode_{}'.format(episode_idx))
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save policy parameters
        model_path = os.path.join(directory, 'policy_parameters.pkl')
        with open(model_path, 'wb') as handle:
            pickle.dump(policy_parameters, handle)
        wandb.save(model_path, wandb.run.dir)

        # Save bnn_model
        bnn_path = os.path.join(directory, 'bnn.pkl')
        with open(bnn_path, 'wb') as handle:
            pickle.dump(bnn_model, handle)
        wandb.save(bnn_path, wandb.run.dir)

        # Save true_data_buffer
        true_data_buffer_path = os.path.join(directory, 'true_data_buffer.pkl')
        with open(true_data_buffer_path, 'wb') as handle:
            pickle.dump(true_data_buffer_state, handle)
        wandb.save(true_data_buffer_path, wandb.run.dir)

        # Save init_states_buffer
        init_states_buffer_path = os.path.join(directory, 'init_states_buffer.pkl')
        with open(init_states_buffer_path, 'wb') as handle:
            pickle.dump(init_states_buffer_state, handle)
        wandb.save(init_states_buffer_path, wandb.run.dir)
        print(f'Saving completed for episode {episode_idx}')

    def run_episodes(self, num_episodes: int, key: chex.PRNGKey):
        key, key_init_buffer, key_init_buffer_init_states = jr.split(key, 3)
        true_buffer_state = self.true_data_buffer_state
        init_states_buffer_state = self.init_states_buffer.init(key_init_buffer_init_states)
        bnn_model = self.bnn_model
        for episode in range(0, num_episodes):
            print(f"Episode {episode}")
            key, key_do_episode = jr.split(key)
            bnn_model, true_buffer_state, init_states_buffer_state = self.do_episode(
                bnn_model=bnn_model,
                true_buffer_state=true_buffer_state,
                init_states_buffer_state=init_states_buffer_state,
                key=key_do_episode,
                episode_idx=episode)


if __name__ == '__main__':
    """
    This is a set uof typical kwargs for SAC
    NUM_ENV_STEPS_BETWEEN_UPDATES = 16
    NUM_ENVS = 64
    SAC_KWARGS = dict(num_timesteps=1_000_000,
                      num_evals=20,
                      reward_scaling=10,
                      episode_length=50,
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
                      num_eval_envs=1,
                      max_replay_size=5 * 10 ** 4,
                      min_replay_size=2 ** 11,
                      policy_hidden_layer_sizes=(64, 64),
                      critic_hidden_layer_sizes=(64, 64),
                      normalize_observations=True,
                      deterministic_eval=True,
                      wandb_logging=True)
    """

    num_env_steps_between_updates = 16
    num_envs = 64

    sac_kwargs = dict(num_timesteps=1_000_000,
                      num_evals=20,
                      reward_scaling=10,
                      episode_length=50,
                      action_repeat=1,
                      discounting=0.99,
                      lr_policy=3e-4,
                      lr_alpha=3e-4,
                      lr_q=3e-4,
                      num_envs=num_envs,
                      batch_size=64,
                      grad_updates_per_step=num_env_steps_between_updates * num_envs,
                      num_env_steps_between_updates=num_env_steps_between_updates,
                      tau=0.005,
                      wd_policy=0,
                      wd_q=0,
                      wd_alpha=0,
                      num_eval_envs=1,
                      max_replay_size=5 * 10 ** 4,
                      min_replay_size=2 ** 11,
                      policy_hidden_layer_sizes=(64, 64),
                      critic_hidden_layer_sizes=(64, 64),
                      normalize_observations=True,
                      deterministic_eval=True,
                      wandb_logging=True)

    ENCODE_ANGLE = True
    ctrl_cost_weight = 0.005
    seed = 0
    num_frame_stack = 3
    gym_env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                          action_delay=0.07,
                          use_tire_model=True,
                          use_obs_noise=True,
                          ctrl_cost_weight=ctrl_cost_weight,
                          )

    x_dim = gym_env.dim_state[0]
    u_dim = gym_env.dim_action[0]
    bnn = BNN_SVGD(x_dim + num_frame_stack * u_dim + u_dim,
                   x_dim,
                   rng_key=jr.PRNGKey(seed),
                   num_train_steps=20000,
                   bandwidth_svgd=10.,
                   likelihood_std=10 * 0.05 * jnp.exp(jnp.array([-3.3170326, -3.7336411, -2.7081904, -2.7081904,
                                                                 -2.7841284, -2.7067015, -1.4446207])),
                   normalize_likelihood_std=True,
                   likelihood_exponent=0.5,
                   learn_likelihood_std=False
                   )
    max_replay_size_true_data_buffer = 10000
    include_aleatoric_noise = True
    car_reward_kwargs = dict(encode_angle=ENCODE_ANGLE,
                             ctrl_cost_weight=ctrl_cost_weight)

    wandb.init(
        project="Race car test MBRL",
        group='test',
    )

    model_based_rl = RealCarRL(gym_env=gym_env,
                               bnn_model=bnn,
                               max_replay_size_true_data_buffer=max_replay_size_true_data_buffer,
                               include_aleatoric_noise=include_aleatoric_noise,
                               car_reward_kwargs=car_reward_kwargs,
                               sac_kwargs=sac_kwargs,
                               num_frame_stack=num_frame_stack,
                               )

    model_based_rl.run_episodes(30, jr.PRNGKey(seed))
