import argparse

import jax.nn
import jax.random as jr
import wandb

from experiments.data_provider import provide_data_and_sim, _RACECAR_NOISE_STD_ENCODED
from grey_box_car_model import GreyBoxSVGDCarModel
from sim_transfer.models import BNN_FSVGD_SimPrior, BNN_FSVGD
from sim_transfer.rl.rl_on_offline_data import RLFromOfflineData
from sim_transfer.sims.simulators import AdditiveSim, PredictStateChangeWrapper, GaussianProcessSim

ENTITY = 'trevenl'


def experiment(horizon_len: int,
               seed: int,
               project_name: str,
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
               use_grey_box: int,
               high_fidelity: int,
               num_measurement_points: int,
               bnn_batch_size: int,
               share_of_x0s_in_sac_buffer: float,
               eval_only_on_init_states: int,
               train_sac_only_from_init_states: int,
               eval_on_all_offline_data: int = 1,
               test_data_ratio: float = 0.2,
               likelihood_exponent: float = 1.0,
               default_num_init_points_to_bs_for_sac_learning=1000,
               data_from_simulation: int = 0,
               num_frame_stack: int = 3,
               bandwidth_svgd: float = 2.0,
               num_epochs: int = 50,
               max_train_steps: int = 100_000
               ):
    bnn_train_steps = min(num_epochs * num_offline_collected_transitions, max_train_steps)

    if not data_from_simulation:
        assert num_frame_stack == 3, "Frame stacking has to be set to 3 if not using simulation data"
    config_dict = dict(use_sim_prior=use_sim_prior,
                       use_grey_box=use_grey_box,
                       high_fidelity=high_fidelity,
                       num_offline_data=num_offline_collected_transitions,
                       share_of_x0s=share_of_x0s_in_sac_buffer,
                       train_sac_only_from_init_states=train_sac_only_from_init_states, )
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
                       high_fidelity=high_fidelity,
                       bnn_batch_size=bnn_batch_size,
                       num_measurement_points=num_measurement_points,
                       test_data_ratio=test_data_ratio,
                       share_of_x0s_in_sac_buffer=share_of_x0s_in_sac_buffer,
                       eval_only_on_init_states=eval_only_on_init_states,
                       eval_on_all_offline_data=eval_on_all_offline_data,
                       train_sac_only_from_init_states=train_sac_only_from_init_states,
                       num_frame_stack=num_frame_stack,
                       bandwidth_svgd=bandwidth_svgd,
                       num_epochs=num_epochs,
                       max_train_steps=max_train_steps,
                       )

    total_config = SAC_KWARGS | config_dict
    group = group_name + '_' + str(likelihood_exponent)
    wandb.init(
        dir='/cluster/scratch/' + ENTITY,
        project=project_name,
        group=group,
        config=total_config,
    )

    # Deal with randomness
    key = jr.PRNGKey(seed)
    key, key_data_seed = jr.split(key, 2)
    int_data_seed = jr.randint(key_data_seed, (), minval=0, maxval=2 ** 13 - 1)
    if data_from_simulation:
        x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
            data_source='racecar_actionstack',
            data_spec={'num_samples_train': num_offline_collected_transitions,
                       'use_hf_sim': bool(high_fidelity),
                       'num_stacked_actions': num_frame_stack},
            data_seed=int(int_data_seed),
        )

    else:
        x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
            data_source='real_racecar_new_actionstack',
            data_spec={'num_samples_train': num_offline_collected_transitions,
                       'use_hf_sim': bool(high_fidelity),
                       'sampling': 'iid',
                       },
            data_seed=int(int_data_seed), )

    # Deal with randomness
    key = jr.PRNGKey(seed)
    key_bnn, key_offline_rl, key_evaluation_trained_bnn, key_evaluation_pretrained_bnn = jr.split(key, 4)

    standard_params = {
        'input_size': sim.input_size,
        'output_size': sim.output_size,
        'rng_key': key_bnn,
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_data': True,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': bool(learnable_likelihood_std),
        'likelihood_exponent': likelihood_exponent,
        'hidden_layer_sizes': [64, 64, 64],
        'normalization_stats': sim.normalization_stats,
        'data_batch_size': bnn_batch_size,
        'hidden_activation': jax.nn.leaky_relu
    }

    if use_sim_prior:
        if high_fidelity:
            outputscales_racecar = [0.008, 0.008, 0.009, 0.009, 0.05, 0.05, 0.20]
        else:
            outputscales_racecar = [0.008, 0.008, 0.01, 0.01, 0.08, 0.08, 0.5]
        sim = AdditiveSim(base_sims=[sim,
                                     GaussianProcessSim(sim.input_size, sim.output_size,
                                                        output_scale=outputscales_racecar,
                                                        length_scale=10.0, consider_only_first_k_dims=None)
                                     ])
        if predict_difference:
            sim = PredictStateChangeWrapper(sim)

        model = BNN_FSVGD_SimPrior(
            **standard_params,
            domain=sim.domain,
            function_sim=sim,
            score_estimator='gp',
            num_train_steps=bnn_train_steps,
            num_f_samples=256,
            bandwidth_svgd=bandwidth_svgd,
            num_measurement_points=num_measurement_points,
        )
    elif use_grey_box:
        if predict_difference:
            raise NotImplementedError("Predicting difference not implemented for grey box model")
        model = GreyBoxSVGDCarModel(
            **standard_params,
            high_fidelity=bool(high_fidelity),
            num_train_steps=bnn_train_steps,
        )
    else:
        model = BNN_FSVGD(
            **standard_params,
            num_train_steps=bnn_train_steps,
            domain=sim.domain,
            bandwidth_svgd=bandwidth_svgd,
        )

    s = share_of_x0s_in_sac_buffer
    num_init_points_to_bs_for_sac_learning = int(num_offline_collected_transitions * s / (1 - s))
    if train_sac_only_from_init_states:
        num_init_points_to_bs_for_sac_learning = default_num_init_points_to_bs_for_sac_learning

    rl_from_offline_data = RLFromOfflineData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        bnn_model=model,
        key=key_offline_rl,
        sac_kwargs=SAC_KWARGS,
        car_reward_kwargs=car_reward_kwargs,
        include_aleatoric_noise=bool(include_aleatoric_noise),
        return_best_policy=bool(best_policy),
        predict_difference=bool(predict_difference),
        test_data_ratio=test_data_ratio,
        eval_bnn_model_on_all_offline_data=False if data_from_simulation else bool(eval_on_all_offline_data),
        eval_sac_only_from_init_states=bool(eval_only_on_init_states),
        num_init_points_to_bs_for_sac_learning=num_init_points_to_bs_for_sac_learning,
        train_sac_only_from_init_states=bool(train_sac_only_from_init_states),
        num_frame_stack=num_frame_stack,
    )
    policy, params, metrics, bnn_model = rl_from_offline_data.prepare_policy_from_offline_data(
        bnn_train_steps=bnn_train_steps,
        return_best_bnn=bool(best_bnn_model))
    rl_from_offline_data.eval_bnn_model_on_test_data(rl_from_offline_data.bnn_model)

    # We evaluate the policy on 100 different initial states and different seeds
    rl_from_offline_data.evaluate_policy(policy, bnn_model, key=key_evaluation_trained_bnn, num_evals=100)
    if data_from_simulation:
        rl_from_offline_data.evaluate_policy_on_the_simulator(policy, key=key_evaluation_pretrained_bnn, num_evals=100)
    else:
        rl_from_offline_data.evaluate_policy(policy, key=key_evaluation_pretrained_bnn, num_evals=100)
    wandb.finish()


def main(args):
    experiment(
        seed=args.seed,
        project_name=args.project_name,
        horizon_len=args.horizon_len,
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
        use_grey_box=args.use_grey_box,
        high_fidelity=args.high_fidelity,
        num_measurement_points=args.num_measurement_points,
        bnn_batch_size=args.bnn_batch_size,
        test_data_ratio=args.test_data_ratio,
        share_of_x0s_in_sac_buffer=args.share_of_x0s_in_sac_buffer,
        eval_only_on_init_states=args.eval_only_on_init_states,
        eval_on_all_offline_data=args.eval_on_all_offline_data,
        likelihood_exponent=args.likelihood_exponent,
        train_sac_only_from_init_states=args.train_sac_only_from_init_states,
        data_from_simulation=args.data_from_simulation,
        num_frame_stack=args.num_frame_stack,
        bandwidth_svgd=args.bandwidth_svgd,
        num_epochs=args.num_epochs,
        max_train_steps=args.max_train_steps,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon_len', type=int, default=200)
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
    parser.add_argument('--num_offline_collected_transitions', type=int, default=20_000)
    parser.add_argument('--use_sim_prior', type=int, default=0)
    parser.add_argument('--use_grey_box', type=int, default=0)
    parser.add_argument('--high_fidelity', type=int, default=0)
    parser.add_argument('--num_measurement_points', type=int, default=8)
    parser.add_argument('--bnn_batch_size', type=int, default=32)
    parser.add_argument('--test_data_ratio', type=float, default=0.0)
    parser.add_argument('--share_of_x0s_in_sac_buffer', type=float, default=0.5)
    parser.add_argument('--eval_only_on_init_states', type=int, default=1)
    parser.add_argument('--eval_on_all_offline_data', type=int, default=1)
    parser.add_argument('--train_sac_only_from_init_states', type=int, default=0)
    parser.add_argument('--likelihood_exponent', type=float, default=1.0)
    parser.add_argument('--data_from_simulation', type=int, default=0)
    parser.add_argument('--num_frame_stack', type=int, default=3)
    parser.add_argument('--bandwidth_svgd', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--max_train_steps', type=int, default=2_000)
    args = parser.parse_args()
    main(args)
