from typing import Callable

import chex
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

import wandb
from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.models.bnn_svgd import BNN_SVGD
from sim_transfer.rl.model_based_rl.learned_system import LearnedCarSystem
from sim_transfer.rl.model_based_rl.utils import split_data
from sim_transfer.sims.envs import RCCarSimEnv

NUM_ENV_STEPS_BETWEEN_UPDATES = 16
NUM_ENVS = 32
SAC_KWARGS = dict(num_timesteps=300_000,
                  num_evals=20,
                  reward_scaling=10,
                  episode_length=30,
                  action_repeat=1,
                  discounting=0.99,
                  lr_policy=3e-4,
                  lr_alpha=3e-4,
                  lr_q=3e-4,
                  num_envs=NUM_ENVS,
                  batch_size=32,
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


class ModelBasedRL:
    def __init__(self,
                 gym_env: RCCarSimEnv,
                 bnn_model: BatchedNeuralNetworkModel = None,
                 max_replay_size_true_data_buffer: int = 10 ** 4,
                 include_aleatoric_noise: bool = True,
                 car_reward_kwargs: dict = None,
                 sac_kwargs: dict = SAC_KWARGS,
                 discounting: chex.Array = jnp.array(0.99),
                 ):
        self.discounting = discounting
        self.car_reward_kwargs = car_reward_kwargs
        self.sac_kwargs = sac_kwargs
        self.include_aleatoric_noise = include_aleatoric_noise
        self.gym_env = gym_env
        self.bnn_model = bnn_model

        self.x_dim = self.gym_env.dim_state[0]
        self.u_dim = self.gym_env.dim_action[0]

        self.dummy_sample = Transition(observation=jnp.zeros(shape=(self.x_dim,)),
                                       action=jnp.zeros(shape=(self.u_dim,)),
                                       reward=jnp.array(0.0),
                                       discount=jnp.array(0.99),
                                       next_observation=jnp.zeros(shape=(self.x_dim,)))

        self.true_data_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)

    def train_policy(self,
                     bnn_model: BatchedNeuralNetworkModel,
                     true_data_buffer_state: ReplayBufferState,
                     key: chex.PRNGKey) -> Callable[[chex.Array], chex.Array]:
        system = LearnedCarSystem(model=bnn_model,
                                  include_noise=self.include_aleatoric_noise,
                                  **self.car_reward_kwargs)
        env = BraxWrapper(system=system,
                          sample_buffer_state=true_data_buffer_state,
                          sample_buffer=self.true_data_buffer,
                          system_params=system.init_params(jr.PRNGKey(0)), )

        sac_trainer = SAC(environment=env, **self.sac_kwargs, )

        params, metrics = sac_trainer.run_training(key=key)
        make_inference_fn = sac_trainer.make_policy

        def policy(x):
            return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]

        return policy

    def simulate_on_true_envs(self,
                              policy: Callable[[chex.Array], chex.Array],
                              key: chex.PRNGKey) -> Transition:
        transitions = []
        obs = self.gym_env.reset(key)
        done = False
        while not done:
            action = policy(obs)
            next_obs, reward, done, info = self.gym_env.step(action)
            transitions.append(Transition(observation=obs,
                                          action=action,
                                          reward=jnp.array(reward),
                                          discount=self.discounting,
                                          next_observation=next_obs))
            obs = next_obs

        concatenated_transitions = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *transitions)
        print('Reward on true system:', jnp.sum(concatenated_transitions.reward))
        return concatenated_transitions

    def train_transition_model(self,
                               true_buffer_state: ReplayBufferState,
                               key: chex.PRNGKey, ) -> BatchedNeuralNetworkModel:
        # Prepare data
        buffer_size = self.true_data_buffer.size(true_buffer_state)
        all_data = true_buffer_state.data[:buffer_size]
        all_transitions = self.true_data_buffer._unflatten_fn(all_data)
        all_obs = all_transitions.observation
        all_actions = all_transitions.action

        all_next_obs = all_transitions.next_observation
        x_all = jnp.concatenate([all_obs, all_actions], axis=-1)
        y_all = all_next_obs - all_obs
        x_train, x_test, y_train, y_test = split_data(x_all, y_all, test_ratio=0.2, seed=42)

        # Train model
        # TODO: Now we are just continuously training the model on the extended data. Should we do this?
        self.bnn_model.fit(x_train=x_train, y_train=y_train, x_eval=x_test, y_eval=y_test, num_steps=20000)
        return self.bnn_model

    def do_episode(self,
                   bnn_model: BatchedNeuralNetworkModel,
                   true_buffer_state: ReplayBufferState,
                   key: chex.PRNGKey):
        # Train policy on current model
        print("Training policy")
        policy = self.train_policy(bnn_model=bnn_model, true_data_buffer_state=true_buffer_state, key=key)
        # Simulate on true envs with policy and collect data
        print("Simulating on true envs")
        transitions = self.simulate_on_true_envs(policy=policy, key=key)
        # Update true data buffer
        print("Updating true data buffer")
        new_true_buffer_state = self.true_data_buffer.insert(true_buffer_state, transitions)
        # Update model
        print("Training bnn model")
        new_bnn_model = self.train_transition_model(true_buffer_state=new_true_buffer_state, key=key)
        return new_bnn_model, new_true_buffer_state

    def run_episodes(self, num_episodes: int, key: chex.PRNGKey):
        key, key_init_buffer = jr.split(key)
        true_buffer_state = self.true_data_buffer.init(key_init_buffer)
        bnn_model = self.bnn_model
        for episode in range(0, num_episodes):
            print(f"Episode {episode}")
            key, key_do_episode = jr.split(key)
            bnn_model, true_buffer_state = self.do_episode(bnn_model=bnn_model,
                                                           true_buffer_state=true_buffer_state,
                                                           key=key_do_episode)


if __name__ == '__main__':
    ENCODE_ANGLE = True
    ctrl_cost_weight = 0.1
    seed = 0
    gym_env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                          action_delay=0.00,
                          use_tire_model=True,
                          use_obs_noise=True,
                          ctrl_cost_weight=ctrl_cost_weight,
                          )

    x_dim = gym_env.dim_state[0]
    u_dim = gym_env.dim_action[0]
    bnn = BNN_SVGD(x_dim + u_dim,
                   x_dim,
                   rng_key=jr.PRNGKey(seed),
                   num_train_steps=20000,
                   bandwidth_svgd=10.,
                   likelihood_std=2 * 0.05 * jnp.exp(jnp.array([-3.3170326, -3.7336411, -2.7081904, -2.7081904,
                                                                -2.7841284, -2.7067015, -1.4446207])),
                   normalize_likelihood_std=True,
                   )
    max_replay_size_true_data_buffer = 10000
    include_aleatoric_noise = True
    car_reward_kwargs = dict(encode_angle=ENCODE_ANGLE,
                             ctrl_cost_weight=ctrl_cost_weight)

    wandb.init(
        project="Race car test MBRL",
        group='test group',
    )

    model_based_rl = ModelBasedRL(gym_env=gym_env,
                                  bnn_model=bnn,
                                  max_replay_size_true_data_buffer=max_replay_size_true_data_buffer,
                                  include_aleatoric_noise=include_aleatoric_noise,
                                  car_reward_kwargs=car_reward_kwargs,
                                  )

    model_based_rl.run_episodes(30, jr.PRNGKey(seed))
