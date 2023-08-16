import argparse

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import wandb
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.sims.car_system import CarSystem


def experiment(num_envs: int,
               net_arch: str,
               seed: int,
               project_name: str,
               batch_size: int,
               max_replay_size: int,
               num_env_steps_between_updates: int,
               ):
    ENCODE_ANGLE = False
    system = CarSystem(encode_angle=ENCODE_ANGLE,
                       action_delay=0.00,
                       use_tire_model=True)

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

    net_arch = {
        "small": dict(policy_hidden_layer_sizes=[64, 64], critic_hidden_layer_sizes=[64, 64]),
        "medium": dict(policy_hidden_layer_sizes=[256, 256], critic_hidden_layer_sizes=[256, 256]),
    }[net_arch]

    sac_config = dict(
        num_envs=num_envs,
        batch_size=batch_size,
        max_replay_size=max_replay_size,
        num_env_steps_between_updates=num_env_steps_between_updates,
        **net_arch,
    )

    group_name = '_'.join(map(lambda x: str(x), list(value for key, value in sac_config.items() if key != 'seed')))

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
        entity="simulation_transfer",
        project=project_name,
        group=group_name,
        config=sac_config,
    )
    params, metrics = sac_trainer.run_training(key=jr.PRNGKey(seed))
    wandb.finish()


def main(args):
    experiment(num_envs=args.num_envs,
               net_arch=args.net_arch,
               seed=args.seed,
               project_name=args.project_name,
               batch_size=args.batch_size,
               max_replay_size=args.max_replay_size,
               num_env_steps_between_updates=args.num_env_steps_between_updates,
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
    args = parser.parse_args()
    main(args)
