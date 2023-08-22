import argparse

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import wandb

from sim_transfer.models.bnn_svgd import BNN_SVGD
from sim_transfer.rl.model_based_rl.main import ModelBasedRL
from sim_transfer.sims.envs import RCCarSimEnv


def experiment(horizon_len: int,
               seed: int,
               project_name: str,
               num_episodes: int,
               bnn_train_steps: int,
               sac_num_env_steps: int,
               learnable_likelihood_std: str
               ):
    config_dict = dict(horizon_len=horizon_len)
    group_name = '_'.join(map(lambda x: str(x), list(value for key, value in config_dict.items() if key != 'seed')))

    NUM_ENV_STEPS_BETWEEN_UPDATES = 16
    NUM_ENVS = 64
    SAC_KWARGS = dict(num_timesteps=sac_num_env_steps,
                      num_evals=20,
                      reward_scaling=10,
                      episode_length=horizon_len,
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
    ENCODE_ANGLE = True
    ctrl_cost_weight = 0.005
    gym_env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                          action_delay=0.00,
                          use_tire_model=True,
                          use_obs_noise=True,
                          ctrl_cost_weight=ctrl_cost_weight,
                          )

    x_dim = gym_env.dim_state[0]
    u_dim = gym_env.dim_action[0]

    learnable_likelihood_std = learnable_likelihood_std == 'yes'
    bnn = BNN_SVGD(x_dim + u_dim,
                   x_dim,
                   rng_key=jr.PRNGKey(seed),
                   num_train_steps=bnn_train_steps,
                   bandwidth_svgd=10.,
                   likelihood_std=10 * 0.05 * jnp.exp(jnp.array([-3.3170326, -3.7336411, -2.7081904, -2.7081904,
                                                                 -2.7841284, -2.7067015, -1.4446207])),
                   normalize_likelihood_std=True,
                   likelihood_exponent=0.5,
                   learn_likelihood_std=learnable_likelihood_std
                   )
    max_replay_size_true_data_buffer = 10000
    include_aleatoric_noise = True
    car_reward_kwargs = dict(encode_angle=ENCODE_ANGLE,
                             ctrl_cost_weight=ctrl_cost_weight)

    total_config = SAC_KWARGS | config_dict
    wandb.init(
        dir='/cluster/scratch/trevenl',
        project=project_name,
        group=group_name,
        config=total_config,
    )

    model_based_rl = ModelBasedRL(gym_env=gym_env,
                                  bnn_model=bnn,
                                  max_replay_size_true_data_buffer=max_replay_size_true_data_buffer,
                                  include_aleatoric_noise=include_aleatoric_noise,
                                  car_reward_kwargs=car_reward_kwargs,
                                  )

    model_based_rl.run_episodes(num_episodes, jr.PRNGKey(seed))

    plt.close('all')
    wandb.finish()


def main(args):
    experiment(
        seed=args.seed,
        project_name=args.project_name,
        horizon_len=args.horizon_len,
        num_episodes=args.num_episodes,
        bnn_train_steps=args.bnn_train_steps,
        sac_num_env_steps=args.sac_num_env_steps,
        learnable_likelihood_std=args.learnable_likelihood_std,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon_len', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=0)
    parser.add_argument('--bnn_train_steps', type=int, default=0)
    parser.add_argument('--sac_num_env_steps', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='RaceCarPPO')
    parser.add_argument('--learnable_likelihood_std', type=str, default='yes')
    args = parser.parse_args()
    main(args)
