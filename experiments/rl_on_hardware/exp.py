import argparse
import pickle

import jax.numpy as jnp
import jax.random as jr
import wandb

from experiments.data_provider import provide_data_and_sim, _RACECAR_NOISE_STD_ENCODED
from sim_transfer.models import BNN_FSVGD_SimPrior, BNN_SVGD
from sim_transfer.rl.model_based_rl.rl_on_hardware import RealCarRL
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.simulators import AdditiveSim, PredictStateChangeWrapper, GaussianProcessSim


def experiment(delay: float = 0.07,
               margin_factor: float = 20.0,
               bnn_train_steps: int = 20_000,
               predict_difference: int = 1,
               learnable_likelihood_std: int = 1,
               num_frame_stack: int = 1,
               use_sim_prior: int = 1,
               max_replay_size_true_data_buffer: int = 10 ** 4,
               include_aleatoric_noise: int = 1,
               sac_num_env_steps: int = 10_000,
               horizon_len: int = 100,
               reset_bnn: int = 1,
               best_bnn_model: int = 1,
               best_policy: int = 1,
               seed: int = 0,
               num_offline_collected_transitions: int = 400,
               ):
    with open('transitions.pkl', 'rb') as f:
        transitions = pickle.load(f)

    ENCODE_ANGLE = True

    x_dim = 7 if ENCODE_ANGLE else 6
    u_dim = 2

    ctrl_cost_weight = 0.005
    gym_env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                          action_delay=delay,
                          use_tire_model=True,
                          use_obs_noise=True,
                          ctrl_cost_weight=ctrl_cost_weight,
                          margin_factor=margin_factor,
                          )

    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(data_source='real_racecar_new_actionstack',
                                                                 data_spec={
                                                                     'num_samples_train': num_offline_collected_transitions})

    """
    Make transtions out of x_train and y_train
    """
    assert x_train.shape[-1] == x_dim + u_dim + num_frame_stack * u_dim

    standard_params = {
        'input_size': sim.input_size,
        'output_size': sim.output_size,
        'rng_key': jr.PRNGKey(seed),
        'likelihood_std': _RACECAR_NOISE_STD_ENCODED,
        'normalize_data': True,
        'normalize_likelihood_std': True,
        'learn_likelihood_std': bool(learnable_likelihood_std),
        'normalization_stats': sim.normalization_stats,
        'likelihood_exponent': 0.5,
        'hidden_layer_sizes': [64, 64, 64],
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
        model = BNN_FSVGD_SimPrior(
            **standard_params,
            domain=sim.domain,
            function_sim=sim,
            score_estimator='gp',
            data_batch_size=32,
            num_train_steps=bnn_train_steps,
            num_f_samples=256,
            bandwidth_svgd=1.0
        )
    else:
        model = BNN_SVGD(
            **standard_params,
            data_batch_size=32,
            num_train_steps=bnn_train_steps,
        )

    #################################################################################################################
    #################################################################################################################

    car_reward_kwargs = dict(encode_angle=ENCODE_ANGLE,
                             ctrl_cost_weight=ctrl_cost_weight,
                             margin_factor=margin_factor)

    print('Using sim prior:', bool(use_sim_prior))

    #################################################################################################################
    #################################################################################################################

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

    wandb.init(
        dir='/cluster/scratch/trevenl',
        project='RealRaceCarRLTest',
        group='test',
    )

    real_car_rl = RealCarRL(
        gym_env=gym_env,
        bnn_model=model,
        offline_data=transitions,
        max_replay_size_true_data_buffer=max_replay_size_true_data_buffer,
        include_aleatoric_noise=bool(include_aleatoric_noise),
        car_reward_kwargs=car_reward_kwargs,
        sac_kwargs=SAC_KWARGS,
        discounting=jnp.array(0.99),
        reset_bnn=bool(reset_bnn),
        return_best_bnn=bool(best_bnn_model),
        return_best_policy=bool(best_policy),
        predict_difference=bool(predict_difference),
        bnn_training_test_ratio=0.2,
        num_frame_stack=num_frame_stack,
        key=jr.PRNGKey(seed),
    )

    real_car_rl.run_episodes(num_episodes=10,
                             key=jr.PRNGKey(seed), )


def main(args):
    experiment(
        delay=args.delay,
        margin_factor=args.margin_factor,
        bnn_train_steps=args.bnn_train_steps,
        predict_difference=args.predict_difference,
        learnable_likelihood_std=args.learnable_likelihood_std,
        num_frame_stack=args.num_frame_stack,
        use_sim_prior=args.use_sim_prior,
        max_replay_size_true_data_buffer=args.max_replay_size_true_data_buffer,
        include_aleatoric_noise=args.include_aleatoric_noise,
        sac_num_env_steps=args.sac_num_env_steps,
        horizon_len=args.horizon_len,
        reset_bnn=args.reset_bnn,
        best_bnn_model=args.best_bnn_model,
        best_policy=args.best_policy,
        seed=args.seed,
        num_offline_collected_transitions=args.num_offline_collected_transitions,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delay', type=float, default=0.07)
    parser.add_argument('--margin_factor', type=float, default=20.0)
    parser.add_argument('--bnn_train_steps', type=int, default=50_000)
    parser.add_argument('--predict_difference', type=int, default=1)
    parser.add_argument('--learnable_likelihood_std', type=int, default=1)
    parser.add_argument('--num_frame_stack', type=int, default=3)
    parser.add_argument('--use_sim_prior', type=int, default=0)
    parser.add_argument('--max_replay_size_true_data_buffer', type=int, default=10 ** 4)
    parser.add_argument('--include_aleatoric_noise', type=int, default=1)
    parser.add_argument('--sac_num_env_steps', type=int, default=1_000_000)
    parser.add_argument('--horizon_len', type=int, default=100)
    parser.add_argument('--reset_bnn', type=int, default=1)
    parser.add_argument('--best_bnn_model', type=int, default=1)
    parser.add_argument('--best_policy', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_offline_collected_transitions', type=int, default=400)
    args = parser.parse_args()
    main(args)
