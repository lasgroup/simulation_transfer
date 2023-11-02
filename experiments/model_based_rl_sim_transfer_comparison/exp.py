import argparse

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import wandb

from experiments.data_provider import _RACECAR_NOISE_STD_ENCODED
from sim_transfer.models import BNN_SVGD, BNN_FSVGD_SimPrior
from sim_transfer.rl.model_based_rl.rl_on_simulator import ModelBasedRL
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.simulators import RaceCarSim, PredictStateChangeWrapper


def experiment(horizon_len: int,
               seed: int,
               project_name: str,
               num_episodes: int,
               bnn_train_steps: int,
               sac_num_env_steps: int,
               learnable_likelihood_std: str,
               reset_bnn: str,
               use_sim_prior: int,
               include_aleatoric_noise: int,
               best_bnn_model: int,
               best_policy: int,
               margin_factor: float,
               predict_difference: int,
               num_frame_stack: int,
               delay: float = 0.00,
               ):
    config_dict = dict(horizon_len=horizon_len,
                       seed=seed,
                       ll_std=learnable_likelihood_std,
                       use_sim_prior=use_sim_prior,
                       bnn_s=bnn_train_steps,
                       sac_s=sac_num_env_steps,
                       bnn_best=best_bnn_model,
                       policy_best=best_policy,
                       pred_dif=predict_difference,
                       fs=num_frame_stack,
                       delay=delay,
                       )
    group_name = '_'.join(list(str(key) + '=' + str(value) for key, value in config_dict.items() if key != 'seed'))

    config_dict = dict(horizon_len=horizon_len,
                       seed=seed,
                       num_episodes=num_episodes,
                       bnn_train_steps=bnn_train_steps,
                       sac_num_env_steps=sac_num_env_steps,
                       ll_std=learnable_likelihood_std,
                       reset_bnn=reset_bnn,
                       use_sim_prior=use_sim_prior,
                       best_bnn_model=best_bnn_model,
                       best_policy=best_policy,
                       margin_factor=margin_factor,
                       predict_difference=predict_difference,
                       num_frame_stack=num_frame_stack,
                       delay=delay,
                       )

    NUM_ENV_STEPS_BETWEEN_UPDATES = 16
    NUM_ENVS = 64
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
    ENCODE_ANGLE = True
    ctrl_cost_weight = 0.005
    gym_env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                          action_delay=delay,
                          use_tire_model=True,
                          use_obs_noise=True,
                          ctrl_cost_weight=ctrl_cost_weight,
                          margin_factor=margin_factor,
                          )

    x_dim = gym_env.dim_state[0]
    u_dim = gym_env.dim_action[0]

    """ Setup a neural network """

    # TODO: Jonas make a wrapper for sim, so that it takes num_frame_stack into account
    _sim = RaceCarSim(encode_angle=True, use_blend=True)
    if predict_difference:
        sim = PredictStateChangeWrapper(_sim)
    else:
        sim = _sim
    learn_std = learnable_likelihood_std == 'yes'

    standard_model_params = {
        'input_size': x_dim + u_dim * num_frame_stack + u_dim,
        'output_size': x_dim,
        'rng_key': jr.PRNGKey(234234345),
        # 'normalization_stats': sim.normalization_stats, TODO: Jonas: adjust sim for normalization stats
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': learn_std,
        'likelihood_exponent': 0.5,
        'hidden_layer_sizes': [64, 64, 64],
        'data_batch_size': 32,
    }

    print('Using sim prior:', bool(use_sim_prior))

    if use_sim_prior:
        bnn = BNN_FSVGD_SimPrior(domain=sim.domain,
                                 function_sim=sim,
                                 num_measurement_points=16,
                                 num_f_samples=512,
                                 score_estimator='gp',
                                 **standard_model_params,
                                 num_train_steps=bnn_train_steps
                                 )
    else:
        bnn = BNN_SVGD(**standard_model_params,
                       bandwidth_svgd=1.0,
                       num_train_steps=bnn_train_steps)

    max_replay_size_true_data_buffer = 10000
    include_aleatoric_noise = include_aleatoric_noise == 1
    car_reward_kwargs = dict(encode_angle=ENCODE_ANGLE,
                             ctrl_cost_weight=ctrl_cost_weight,
                             margin_factor=margin_factor)

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
                                  reset_bnn=reset_bnn == 'yes',
                                  return_best_bnn=bool(best_bnn_model),
                                  return_best_policy=bool(best_policy),
                                  sac_kwargs=SAC_KWARGS,
                                  discounting=jnp.array(0.99),
                                  predict_difference=bool(predict_difference),
                                  num_frame_stack=num_frame_stack,
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
        reset_bnn=args.reset_bnn,
        use_sim_prior=args.use_sim_prior,
        include_aleatoric_noise=args.include_aleatoric_noise,
        best_bnn_model=args.best_bnn_model,
        best_policy=args.best_policy,
        margin_factor=args.margin_factor,
        predict_difference=args.predict_difference,
        num_frame_stack=args.num_frame_stack,
        delay=args.delay,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon_len', type=int, default=200)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--bnn_train_steps', type=int, default=10_000)
    parser.add_argument('--sac_num_env_steps', type=int, default=10_000)
    parser.add_argument('--project_name', type=str, default='RaceCarPPO')
    parser.add_argument('--learnable_likelihood_std', type=str, default='yes')
    parser.add_argument('--reset_bnn', type=str, default='yes')
    parser.add_argument('--use_sim_prior', type=int, default=0)
    parser.add_argument('--include_aleatoric_noise', type=int, default=1)
    parser.add_argument('--best_bnn_model', type=int, default=1)
    parser.add_argument('--best_policy', type=int, default=0)
    parser.add_argument('--margin_factor', type=float, default=20.0)
    parser.add_argument('--predict_difference', type=int, default=0)
    parser.add_argument('--num_frame_stack', type=int, default=2)
    parser.add_argument('--delay', type=float, default=0.07)
    args = parser.parse_args()
    main(args)
