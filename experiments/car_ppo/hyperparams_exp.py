import argparse

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import wandb
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from mbpo.optimizers.policy_optimizers.ppo.ppo import PPO
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.sims.car_system import CarSystem


def experiment(num_envs: int,
               lr: float,
               entropy_cost: float,
               unroll_length: int,
               batch_size: int,
               num_minibatches: int,
               num_updates_per_batch: int,
               net_arch: str,
               seed: int,
               project_name: str):
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

    ppo_config = dict(
        seed=seed,
        num_envs=num_envs,
        lr=lr,
        entropy_cost=entropy_cost,
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        **net_arch,
    )

    group_name = '_'.join(map(lambda x: str(x), list(value for key, value in ppo_config.items() if key != 'seed')))

    ppo_trainer = PPO(
        environment=env,
        num_timesteps=10_000_000,
        episode_length=200,
        action_repeat=1,
        num_eval_envs=1,
        wd=0,
        discounting=0.99,
        num_evals=20,
        reward_scaling=10,
        clipping_epsilon=0.3,
        gae_lambda=0.95,
        normalize_observations=True,
        normalize_advantage=True,
        deterministic_eval=True,
        wandb_logging=True,
        **ppo_config,
    )

    wandb.init(
        dir='/cluster/scratch/trevenl',
        entity="simulation_transfer",
        project=project_name,
        group=group_name,
        config=ppo_config,
    )
    params, metrics = ppo_trainer.run_training(key=jr.PRNGKey(seed))
    wandb.finish()


def main(args):
    experiment(num_envs=args.num_envs,
               lr=args.lr,
               entropy_cost=args.entropy_cost,
               unroll_length=args.unroll_length,
               batch_size=args.batch_size,
               num_minibatches=args.num_minibatches,
               num_updates_per_batch=args.num_updates_per_batch,
               net_arch=args.net_arch,
               seed=args.seed,
               project_name=args.project_name,
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--unroll_length', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=8)
    parser.add_argument('--net_arch', type=str, default='small')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='RaceCarPPO')
    args = parser.parse_args()
    main(args)
