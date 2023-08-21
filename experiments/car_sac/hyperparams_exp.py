import argparse

import chex
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax import vmap
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.sims.car_system import CarSystem
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.util import plot_rc_trajectory


def experiment(num_envs: int,
               net_arch: str,
               seed: int,
               project_name: str,
               batch_size: int,
               max_replay_size: int,
               num_env_steps_between_updates: int,
               target_entropy: float,
               ):
    ENCODE_ANGLE = False
    system = CarSystem(encode_angle=ENCODE_ANGLE,
                       action_delay=0.00,
                       use_tire_model=True,
                       ctrl_cost_weight=0.005)

    # Create replay buffer
    num_init_states = 1000
    keys = jr.split(jr.PRNGKey(seed), num_init_states)
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
                      system_params=system.init_params(jr.PRNGKey(seed)), )

    net_arch = {
        "small": dict(policy_hidden_layer_sizes=[64, 64], critic_hidden_layer_sizes=[64, 64]),
        "medium": dict(policy_hidden_layer_sizes=[256, 256], critic_hidden_layer_sizes=[256, 256]),
    }[net_arch]

    sac_config = dict(
        num_envs=num_envs,
        batch_size=batch_size,
        max_replay_size=max_replay_size,
        num_env_steps_between_updates=num_env_steps_between_updates,
        target_entropy=target_entropy,
        **net_arch,
    )

    group_name = '_'.join(map(lambda x: str(x), list(value for key, value in sac_config.items() if key != 'seed')))

    discounting = 0.99
    sac_trainer = SAC(
        environment=env,
        num_timesteps=2_000_000,
        num_evals=20,
        reward_scaling=10,
        episode_length=300,
        action_repeat=1,
        discounting=discounting,
        lr_policy=3e-4,
        lr_alpha=3e-4,
        lr_q=3e-4,
        grad_updates_per_step=num_env_steps_between_updates * num_envs,
        min_replay_size=2 ** 11,
        num_eval_envs=1,
        tau=0.005,
        wd_policy=0,
        wd_q=0,
        wd_alpha=0,
        normalize_observations=True,
        deterministic_eval=True,
        wandb_logging=True,
        **sac_config
    )

    wandb.init(
        dir='/cluster/scratch/trevenl',
        project=project_name,
        group=group_name,
        config=sac_config,
    )
    params, metrics = sac_trainer.run_training(key=jr.PRNGKey(seed))

    def policy(x):
        return sac_trainer.make_policy(params, deterministic=True)(x, jr.PRNGKey(0))[0]

    gym_env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                          action_delay=0.00,
                          use_tire_model=True,
                          use_obs_noise=True,
                          ctrl_cost_weight=0.005,
                          )

    def simulate_on_true_envs(key: chex.PRNGKey) -> Transition:
        transitions = []
        obs = gym_env.reset(key)
        done = False
        while not done:
            action = policy(obs)
            next_obs, reward, done, info = gym_env.step(action)
            transitions.append(Transition(observation=obs,
                                          action=action,
                                          reward=jnp.array(reward),
                                          discount=discounting,
                                          next_observation=next_obs))
            obs = next_obs

        concatenated_transitions = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *transitions)
        reward_on_true_system = jnp.sum(concatenated_transitions.reward)
        print('Reward on true system:', reward_on_true_system)
        wandb.log({'reward_on_true_system': reward_on_true_system})

        fig, axes = plot_rc_trajectory(concatenated_transitions.next_observation,
                                       concatenated_transitions.action, encode_angle=ENCODE_ANGLE,
                                       show=False)
        wandb.log({'True_trajectory_path': wandb.Image(fig)})
        plt.close('all')
        return concatenated_transitions

    concatenated_transitions = simulate_on_true_envs(jr.PRNGKey(0))

    fig, axes = plot_rc_trajectory(concatenated_transitions.next_observation,
                                   concatenated_transitions.action, encode_angle=ENCODE_ANGLE,
                                   show=False)
    wandb.log({'True_trajectory_path': wandb.Image(fig)})
    plt.close('all')

    wandb.finish()


def main(args):
    experiment(num_envs=args.num_envs,
               net_arch=args.net_arch,
               seed=args.seed,
               project_name=args.project_name,
               batch_size=args.batch_size,
               max_replay_size=args.max_replay_size,
               num_env_steps_between_updates=args.num_env_steps_between_updates,
               target_entropy=args.target_entropy,
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--net_arch', type=str, default='small')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='RaceCarPPO')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_replay_size', type=int, default=10 ** 6)
    parser.add_argument('--num_env_steps_between_updates', type=int, default=32)
    parser.add_argument('--target_entropy', type=float, default=-10.0)
    args = parser.parse_args()
    main(args)
