import argparse

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import wandb

from experiments.data_provider import _RACECAR_NOISE_STD_ENCODED
from sim_transfer.models import BNN_FSVGD_SimPrior, BNN_FSVGD
from sim_transfer.rl.model_based_rl.rl_on_simulator import ModelBasedRL
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.simulators import AdditiveSim, PredictStateChangeWrapper, GaussianProcessSim
from sim_transfer.sims.simulators import RaceCarSim, StackedActionSimWrapper

ENCODE_ANGLE = True
ENTITY = 'trevenl'
PRIORS = {'none', 'high_fidelity', 'low_fidelity'}


def experiment(horizon_len: int,
               seed: int,
               project_name: str,
               num_episodes: int,
               bnn_train_steps: int,
               sac_num_env_steps: int,
               learnable_likelihood_std: int,
               reset_bnn: int,
               sim_prior: str,
               include_aleatoric_noise: int,
               best_bnn_model: int,
               best_policy: int,
               predict_difference: int,
               margin_factor: float,
               ctrl_cost_weight: float = 0.005,
               num_stacked_actions: int = 3,
               delay: float = 3 / 30,
               max_replay_size_true_data_buffer: int = 10000,
               likelihood_exponent: float = 0.5,
               data_batch_size: int = 32,
               bandwidth_svgd: float = 0.2,
               length_scale_aditive_sim_gp: float = 10.0,
               num_f_samples: int = 512,
               num_measurement_points: int = 16,
               ):
    assert sim_prior in PRIORS, f'Invalid sim prior: {sim_prior}'

    """Setup key"""
    key = jr.PRNGKey(seed)
    key_sim, *keys = jr.split(key, 2)

    """Setup group name"""
    group_name_config = dict(horizon_len=horizon_len,
                             sim_prior=sim_prior,
                             )
    group_name = '_'.join(list(str(key) + '=' + str(value) for key, value in group_name_config.items()))

    """Setup car reward kwargs"""
    car_reward_kwargs = dict(encode_angle=ENCODE_ANGLE,
                             ctrl_cost_weight=ctrl_cost_weight,
                             margin_factor=margin_factor)

    """Setup config dict"""
    config_dict = dict(horizon_len=horizon_len,
                       seed=seed,
                       num_episodes=num_episodes,
                       bnn_train_steps=bnn_train_steps,
                       sac_num_env_steps=sac_num_env_steps,
                       ll_std=learnable_likelihood_std,
                       reset_bnn=reset_bnn,
                       sim_prior=sim_prior,
                       include_aleatoric_noise=include_aleatoric_noise,
                       best_bnn_model=best_bnn_model,
                       best_policy=best_policy,
                       margin_factor=margin_factor,
                       predict_difference=predict_difference,
                       num_stacked_actions=num_stacked_actions,
                       delay=delay,
                       max_replay_size_true_data_buffer=max_replay_size_true_data_buffer,
                       car_reward_kwargs=car_reward_kwargs,
                       likelihood_exponent=likelihood_exponent,
                       data_batch_size=data_batch_size,
                       )

    """Setup SAC config dict"""
    num_env_steps_between_updates = 16
    num_envs = 64
    sac_kwargs = dict(num_timesteps=sac_num_env_steps,
                      num_evals=20,
                      reward_scaling=10,
                      episode_length=horizon_len,
                      episode_length_eval=2 * horizon_len,
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
                      num_eval_envs=2 * num_envs,
                      max_replay_size=5 * 10 ** 4,
                      min_replay_size=2 ** 11,
                      policy_hidden_layer_sizes=(64, 64),
                      critic_hidden_layer_sizes=(64, 64),
                      normalize_observations=True,
                      deterministic_eval=True,
                      wandb_logging=True)

    """Setup gym-like environment"""
    gym_env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                          action_delay=delay,
                          use_tire_model=True,
                          use_obs_noise=True,
                          ctrl_cost_weight=ctrl_cost_weight,
                          margin_factor=margin_factor,
                          )

    total_config = sac_kwargs | config_dict
    wandb.init(
        dir='/cluster/scratch/' + ENTITY,
        project=project_name,
        group=group_name,
        config=total_config,
    )

    """ Setup BNN"""
    sim = RaceCarSim(encode_angle=True, use_blend=sim_prior == 'high_fidelity', car_id=2)
    if num_stacked_actions > 0:
        sim = StackedActionSimWrapper(sim, num_stacked_actions=num_stacked_actions, action_size=2)
    if predict_difference:
        sim = PredictStateChangeWrapper(sim)

    standard_params = {
        'input_size': sim.input_size,
        'output_size': sim.output_size,
        'rng_key': key_sim,
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_data': True,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': bool(learnable_likelihood_std),
        'likelihood_exponent': likelihood_exponent,
        'hidden_layer_sizes': [64, 64, 64],
        'normalization_stats': sim.normalization_stats,
        'data_batch_size': data_batch_size,
        'hidden_activation': jax.nn.leaky_relu
    }

    if sim_prior == 'none':
        bnn = BNN_FSVGD(
            **standard_params,
            num_train_steps=bnn_train_steps,
            domain=sim.domain,
            bandwidth_svgd=bandwidth_svgd,
        )
    else:
        if sim_prior == 'high_fidelity':
            outputscales_racecar = [0.008, 0.008, 0.009, 0.009, 0.05, 0.05, 0.20]
        elif sim_prior == 'low_fidelity':
            outputscales_racecar = [0.008, 0.008, 0.01, 0.01, 0.08, 0.08, 0.5]
        else:
            raise ValueError(f'Invalid sim prior: {sim_prior}')

        sim = AdditiveSim(base_sims=[sim,
                                     GaussianProcessSim(sim.input_size, sim.output_size,
                                                        output_scale=outputscales_racecar,
                                                        length_scale=length_scale_aditive_sim_gp,
                                                        consider_only_first_k_dims=None)
                                     ])

        bnn = BNN_FSVGD_SimPrior(
            **standard_params,
            domain=sim.domain,
            function_sim=sim,
            score_estimator='gp',
            num_train_steps=bnn_train_steps,
            num_f_samples=num_f_samples,
            bandwidth_svgd=bandwidth_svgd,
            num_measurement_points=num_measurement_points,
        )

    model_based_rl = ModelBasedRL(gym_env=gym_env,
                                  bnn_model=bnn,
                                  max_replay_size_true_data_buffer=max_replay_size_true_data_buffer,
                                  include_aleatoric_noise=bool(include_aleatoric_noise),
                                  car_reward_kwargs=car_reward_kwargs,
                                  reset_bnn=bool(reset_bnn),
                                  return_best_bnn=bool(best_bnn_model),
                                  return_best_policy=bool(best_policy),
                                  sac_kwargs=sac_kwargs,
                                  predict_difference=bool(predict_difference),
                                  num_stacked_actions=num_stacked_actions,
                                  )

    model_based_rl.run_episodes(num_episodes, jr.PRNGKey(seed))

    plt.close('all')
    wandb.finish()


def main(args):
    experiment(
        horizon_len=args.horizon_len,
        seed=args.seed,
        project_name=args.project_name,
        num_episodes=args.num_episodes,
        bnn_train_steps=args.bnn_train_steps,
        sac_num_env_steps=args.sac_num_env_steps,
        learnable_likelihood_std=args.learnable_likelihood_std,
        reset_bnn=args.reset_bnn,
        sim_prior=args.sim_prior,
        include_aleatoric_noise=args.include_aleatoric_noise,
        best_bnn_model=args.best_bnn_model,
        best_policy=args.best_policy,
        predict_difference=args.predict_difference,
        margin_factor=args.margin_factor,
        ctrl_cost_weight=args.ctrl_cost_weight,
        num_stacked_actions=args.num_stacked_actions,
        delay=args.delay,
        max_replay_size_true_data_buffer=args.max_replay_size_true_data_buffer,
        likelihood_exponent=args.likelihood_exponent,
        data_batch_size=args.data_batch_size,
        bandwidth_svgd=args.bandwidth_svgd,
        length_scale_aditive_sim_gp=args.length_scale_aditive_sim_gp,
        num_f_samples=args.num_f_samples,
        num_measurement_points=args.num_measurement_points,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon_len', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='RaceCarPPO')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--bnn_train_steps', type=int, default=10_000)
    parser.add_argument('--sac_num_env_steps', type=int, default=10_000)
    parser.add_argument('--learnable_likelihood_std', type=int, default=1)
    parser.add_argument('--reset_bnn', type=int, default=1)
    parser.add_argument('--sim_prior', type=str, default='none')
    parser.add_argument('--include_aleatoric_noise', type=int, default=1)
    parser.add_argument('--best_bnn_model', type=int, default=1)
    parser.add_argument('--best_policy', type=int, default=0)
    parser.add_argument('--predict_difference', type=int, default=0)
    parser.add_argument('--margin_factor', type=float, default=20.0)
    parser.add_argument('--ctrl_cost_weight', type=float, default=0.005)
    parser.add_argument('--num_stacked_actions', type=int, default=3)
    parser.add_argument('--delay', type=float, default=3 / 30)
    parser.add_argument('--max_replay_size_true_data_buffer', type=int, default=10_000)
    parser.add_argument('--likelihood_exponent', type=float, default=0.5)
    parser.add_argument('--data_batch_size', type=int, default=32)
    parser.add_argument('--bandwidth_svgd', type=float, default=0.2)
    parser.add_argument('--length_scale_aditive_sim_gp', type=float, default=10.0)
    parser.add_argument('--num_f_samples', type=int, default=512)
    parser.add_argument('--num_measurement_points', type=int, default=16)

    args = parser.parse_args()
    main(args)
