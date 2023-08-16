from datetime import datetime

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax import jit
from jax.lax import scan
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.sims.car_system import CarSystem
from sim_transfer.sims.util import plot_rc_trajectory

ENCODE_ANGLE = False
system = CarSystem(encode_angle=ENCODE_ANGLE,
                   action_delay=0.00,
                   use_tire_model=True,
                   use_obs_noise=True,
                   )

# Create replay buffer
init_sys_state = system.reset(key=jr.PRNGKey(0))

dummy_sample = Transition(observation=init_sys_state.x_next,
                          action=jnp.zeros(shape=(system.u_dim,)),
                          reward=init_sys_state.reward,
                          discount=jnp.array(0.99),
                          next_observation=init_sys_state.x_next)

sampling_buffer = UniformSamplingQueue(max_replay_size=1,
                                       dummy_data_sample=dummy_sample,
                                       sample_batch_size=1)

sampling_buffer_state = sampling_buffer.init(jr.PRNGKey(0))
sampling_buffer_state = sampling_buffer.insert(sampling_buffer_state,
                                               jtu.tree_map(lambda x: x[None, ...], dummy_sample))

# Create brax environment
env = BraxWrapper(system=system,
                  sample_buffer_state=sampling_buffer_state,
                  sample_buffer=sampling_buffer,
                  system_params=system.init_params(jr.PRNGKey(0)), )

state = jit(env.reset)(rng=jr.PRNGKey(0))

num_env_steps_between_updates = 16
num_envs = 32

sac_trainer = SAC(
    environment=env,
    num_timesteps=300_000,
    num_evals=20,
    reward_scaling=10,
    episode_length=200,
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
    min_replay_size=2 ** 11,
    policy_hidden_layer_sizes=(64, 64),
    critic_hidden_layer_sizes=(64, 64),
    normalize_observations=True,
    deterministic_eval=True,
    wandb_logging=False,
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


system_state_init = system.reset(key=jr.PRNGKey(0))
x_init = system_state_init.x_next
system_params = system_state_init.system_params


def step(system_state, _):
    u = policy(system_state.x_next)
    next_sys_state = system.step(system_state.x_next, u, system_state.system_params)
    return next_sys_state, (system_state.x_next, u, next_sys_state.reward)


horizon = 200
x_last, trajectory = scan(step, system_state_init, None, length=horizon)

plt.plot(trajectory[0], label='Xs')
plt.plot(trajectory[1], label='Us')
plt.plot(trajectory[2], label='Rewards')
plt.legend()
plt.show()

traj = trajectory[0]
actions = trajectory[1]

plot_rc_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)
