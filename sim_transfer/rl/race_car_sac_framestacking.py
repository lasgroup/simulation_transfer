from datetime import datetime

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax import jit, vmap
from jax.lax import scan
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.sims.car_system import CarSystem, FrameStackWrapper
from sim_transfer.sims.util import plot_rc_trajectory

# import os
# os.environ['JAX_LOG_COMPILES'] = '1'

ENCODE_ANGLE = True
_system = CarSystem(encode_angle=ENCODE_ANGLE,
                    action_delay=0.09,
                    use_tire_model=True,
                    use_obs_noise=True,
                    ctrl_cost_weight=0.005,
                    margin_factor=20,
                    )

# Here we create framestacking wrapper
num_frame_stack = 3
system = FrameStackWrapper(_system, num_frame_stack)

# Create replay buffer
num_init_states = 10
keys = jr.split(jr.PRNGKey(0), num_init_states)
init_sys_state = vmap(system._system.reset)(key=keys)

init_us = jnp.zeros(shape=(num_init_states, _system.u_dim * num_frame_stack))

# Here we need to repeat observations
init_samples = Transition(observation=jnp.concatenate([init_sys_state.x_next, init_us], axis=-1),
                          action=jnp.zeros(shape=(num_init_states, system.u_dim,)),
                          reward=init_sys_state.reward,
                          discount=0.99 * jnp.ones(shape=(num_init_states,)),
                          next_observation=jnp.concatenate([init_sys_state.x_next, init_us], axis=-1))

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

state = jit(env.reset)(rng=jr.PRNGKey(0))

num_env_steps_between_updates = 1
num_envs = 32
horizon = 200

sac_trainer = SAC(
    environment=env,
    num_timesteps=1_000_000,
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
    max_replay_size=10 ** 5,
    min_replay_size=10 ** 3,
    policy_hidden_layer_sizes=(64, 64),
    critic_hidden_layer_sizes=(64, 64),
    normalize_observations=True,
    deterministic_eval=True,
    wandb_logging=False,
    return_best_model=True,
)

max_y = 0
min_y = -100

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    # plt.xlim([0, sac_trainer.num_timesteps])
    # plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()


params, metrics = sac_trainer.run_training(key=jr.PRNGKey(0), progress_fn=progress)

make_inference_fn = sac_trainer.make_policy

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')


def policy(x):
    return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]


test_system = FrameStackWrapper(_system, num_frame_stack)

system_state_init = system._system.reset(key=jr.PRNGKey(0))

init_u = jnp.zeros(shape=(_system.u_dim * num_frame_stack,))
x_init = jnp.concatenate([system_state_init.x_next, init_u], axis=-1)

system_state_init = system_state_init.replace(x_next=x_init)
system_params = system_state_init.system_params


def step(system_state, _):
    u = policy(system_state.x_next)
    next_sys_state = test_system.step(system_state.x_next, u, system_state.system_params)
    return next_sys_state, (system_state.x_next[:system._system.x_dim], u, next_sys_state.reward)


x_last, trajectory = scan(step, system_state_init, None, length=horizon)

plt.plot(trajectory[0], label='Xs')
plt.plot(trajectory[1], label='Us')
plt.plot(trajectory[2], label='Rewards')
plt.legend()
plt.show()
print('Reward: ', jnp.sum(trajectory[2]))

traj = trajectory[0]
actions = trajectory[1]

plot_rc_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)

SAVE = True

if SAVE:
    import pickle

    with open('params.pkl', 'wb') as file:
        pickle.dump(params, file)
