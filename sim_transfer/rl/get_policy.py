import pickle

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax import vmap
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.sims.car_system import CarSystem

ENCODE_ANGLE = True
system = CarSystem(encode_angle=ENCODE_ANGLE,
                   action_delay=0.00,
                   use_tire_model=True,
                   use_obs_noise=True,
                   ctrl_cost_weight=0.005,
                   )

# Create replay buffer
num_init_states = 500
keys = jr.split(jr.PRNGKey(0), num_init_states)
init_sys_state = vmap(system.reset)(key=keys)

init_samples = Transition(observation=init_sys_state.x_next,
                          action=jnp.zeros(shape=(num_init_states, system.u_dim,)),
                          reward=init_sys_state.reward,
                          discount=0.99 * jnp.ones(shape=(num_init_states,)),
                          next_observation=init_sys_state.x_next)

dummy_sample = jtu.tree_map(lambda x: x[0], init_samples)

sampling_buffer = UniformSamplingQueue(max_replay_size=num_init_states,
                                       dummy_data_sample=dummy_sample,
                                       sample_batch_size=1)

sampling_buffer_state = sampling_buffer.init(jr.PRNGKey(0))

sampling_buffer_state = sampling_buffer.insert(sampling_buffer_state, init_samples)

# Create brax environment
env = BraxWrapper(system=system,
                  sample_buffer_state=sampling_buffer_state,
                  sample_buffer=sampling_buffer,
                  system_params=system.init_params(jr.PRNGKey(0)), )

num_env_steps_between_updates = 32
num_envs = 16
horizon = 300

sac_trainer = SAC(
    target_entropy=-10,
    environment=env,
    num_timesteps=700_000,
    num_evals=20,
    reward_scaling=1,
    episode_length=horizon,
    action_repeat=1,
    discounting=0.99,
    lr_policy=3e-4,
    lr_alpha=3e-4,
    lr_q=3e-4,
    num_envs=num_envs,
    batch_size=32,
    grad_updates_per_step=num_env_steps_between_updates * num_envs,
    num_env_steps_between_updates=num_env_steps_between_updates,
    tau=0.005,
    wd_policy=0,
    wd_q=0,
    wd_alpha=0,
    num_eval_envs=1,
    max_replay_size=5 * 10 ** 4,
    min_replay_size=10 ** 3,
    policy_hidden_layer_sizes=(64, 64),
    critic_hidden_layer_sizes=(64, 64),
    normalize_observations=True,
    deterministic_eval=True,
    wandb_logging=False,
)

make_inference_fn = sac_trainer.make_policy

with open('params.pkl', 'rb') as file:
    params = pickle.load(file)


def policy(x):
    return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]
