import argparse

import jax.random as jr
import wandb

from experiments.data_provider import provide_data_and_sim, _RACECAR_NOISE_STD_ENCODED
from sim_transfer.models import BNN_FSVGD_SimPrior, BNN_SVGD
from sim_transfer.rl.rl_on_offline_data import RLFromOfflineData
from sim_transfer.sims.simulators import AdditiveSim, PredictStateChangeWrapper, GaussianProcessSim


def experiment(horizon_len: int,
               seed: int,
               project_name: str,
               bnn_train_steps: int,
               sac_num_env_steps: int,
               learnable_likelihood_std: str,
               include_aleatoric_noise: int,
               best_bnn_model: int,
               best_policy: int,
               margin_factor: float,
               predict_difference: int,
               ctrl_cost_weight: float,
               ctrl_diff_weight: float,
               num_offline_collected_transitions: int,
               use_sim_prior: int,
               use_sim_normalization_stats: int
               ):
    config_dict = dict(horizon_len=horizon_len,
                       seed=seed,
                       diff_w=ctrl_diff_weight,
                       cost_w=ctrl_cost_weight,
                       num_offline_trans=num_offline_collected_transitions,
                       use_sim_prior=use_sim_prior,
                       use_sim_norm_stats=use_sim_normalization_stats
                       )
    group_name = '_'.join(list(str(key) + '=' + str(value) for key, value in config_dict.items() if key != 'seed'))

    car_reward_kwargs = dict(encode_angle=True,
                             ctrl_cost_weight=ctrl_cost_weight,
                             margin_factor=margin_factor,
                             ctrl_diff_weight=ctrl_diff_weight,
                             )

    NUM_ENV_STEPS_BETWEEN_UPDATES = 16
    NUM_ENVS = 64

    SAC_KWARGS = dict(num_timesteps=sac_num_env_steps,
                      num_evals=20,
                      reward_scaling=10,
                      episode_length=horizon_len,
                      episode_length_eval=200,
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

    config_dict = dict(horizon_len=horizon_len,
                       seed=seed,
                       bnn_train_steps=bnn_train_steps,
                       sac_num_env_steps=sac_num_env_steps,
                       ll_std=learnable_likelihood_std,
                       best_bnn_model=best_bnn_model,
                       best_policy=best_policy,
                       margin_factor=margin_factor,
                       predict_difference=predict_difference,
                       ctrl_diff_weight=ctrl_diff_weight,
                       ctrl_cost_weight=ctrl_cost_weight,
                       num_offline_collected_transitions=num_offline_collected_transitions,
                       use_sim_prior=use_sim_prior,
                       use_sim_normalization_stats=use_sim_normalization_stats
                       )

    total_config = SAC_KWARGS | config_dict
    wandb.init(
        dir='/cluster/scratch/trevenl',
        project=project_name,
        group=group_name,
        config=total_config,
    )

    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(data_source='real_racecar_new_actionstack',
                                                                 data_spec={
                                                                     'num_samples_train': num_offline_collected_transitions})

    standard_params = {
        'input_size': sim.input_size,
        'output_size': sim.output_size,
        'rng_key': jr.PRNGKey(seed),
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_data': True,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': bool(learnable_likelihood_std),
        'likelihood_exponent': 0.5,
        'hidden_layer_sizes': [64, 64, 64],
        'data_batch_size': 128,
    }

    if use_sim_prior:
        outputscales_racecar = [0.007, 0.007, 0.007, 0.007, 0.04, 0.04, 0.18]
        sim = AdditiveSim(base_sims=[sim,
                                     GaussianProcessSim(sim.input_size, sim.output_size,
                                                        output_scale=outputscales_racecar,
                                                        length_scale=10.0, consider_only_first_k_dims=None)
                                     ])
        if predict_difference:
            sim = PredictStateChangeWrapper(sim)
        if use_sim_normalization_stats:
            standard_params['normalization_stats'] = sim.normalization_stats
        model = BNN_FSVGD_SimPrior(
            **standard_params,
            domain=sim.domain,
            function_sim=sim,
            score_estimator='gp',
            num_train_steps=bnn_train_steps,
            num_f_samples=256,
            bandwidth_svgd=1.0
        )
    else:
        if predict_difference:
            sim = PredictStateChangeWrapper(sim)
        if use_sim_normalization_stats:
            standard_params['normalization_stats'] = sim.normalization_stats
        model = BNN_SVGD(
            **standard_params,
            num_train_steps=bnn_train_steps,
        )

    rl_from_offline_data = RLFromOfflineData(
        data_spec={'num_samples_train': num_offline_collected_transitions},
        bnn_model=model,
        key=jr.PRNGKey(seed),
        sac_kwargs=SAC_KWARGS,
        car_reward_kwargs=car_reward_kwargs,
        include_aleatoric_noise=bool(include_aleatoric_noise),
        return_best_policy=bool(best_policy),
        predict_difference=bool(predict_difference),
    )
    policy, params, metrics, bnn_model = rl_from_offline_data.prepare_policy_from_offline_data(
        bnn_train_steps=bnn_train_steps,
        return_best_bnn=bool(best_bnn_model))

    rl_from_offline_data.evaluate_policy(policy, bnn_model, key=jr.PRNGKey(0))
    wandb.finish()


def main(args):
    experiment(
        seed=args.seed,
        project_name=args.project_name,
        horizon_len=args.horizon_len,
        bnn_train_steps=args.bnn_train_steps,
        sac_num_env_steps=args.sac_num_env_steps,
        learnable_likelihood_std=args.learnable_likelihood_std,
        include_aleatoric_noise=args.include_aleatoric_noise,
        best_bnn_model=args.best_bnn_model,
        best_policy=args.best_policy,
        margin_factor=args.margin_factor,
        predict_difference=args.predict_difference,
        ctrl_cost_weight=args.ctrl_cost_weight,
        ctrl_diff_weight=args.ctrl_diff_weight,
        num_offline_collected_transitions=args.num_offline_collected_transitions,
        use_sim_prior=args.use_sim_prior,
        use_sim_normalization_stats=args.use_sim_normalization_stats,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon_len', type=int, default=200)
    parser.add_argument('--bnn_train_steps', type=int, default=2_000)
    parser.add_argument('--sac_num_env_steps', type=int, default=10_000)
    parser.add_argument('--project_name', type=str, default='RaceCarPPO')
    parser.add_argument('--learnable_likelihood_std', type=str, default='yes')
    parser.add_argument('--include_aleatoric_noise', type=int, default=1)
    parser.add_argument('--best_bnn_model', type=int, default=1)
    parser.add_argument('--best_policy', type=int, default=0)
    parser.add_argument('--margin_factor', type=float, default=20.0)
    parser.add_argument('--predict_difference', type=int, default=0)
    parser.add_argument('--ctrl_cost_weight', type=float, default=0.005)
    parser.add_argument('--ctrl_diff_weight', type=float, default=0.01)
    parser.add_argument('--num_offline_collected_transitions', type=int, default=1_000)
    parser.add_argument('--use_sim_prior', type=int, default=1)
    parser.add_argument('--use_sim_normalization_stats', type=int, default=1)
    args = parser.parse_args()
    main(args)
