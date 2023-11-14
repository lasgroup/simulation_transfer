import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'RLComparisonOnSimulatorN2'

applicable_configs = {
    'horizon_len': [2 ** 6],
    'seed': list(range(10)),
    'project_name': [PROJECT_NAME],
    'num_episodes': [20],
    'bnn_train_steps': [40_000, 80_000],
    'sac_num_env_steps': [1_000_000, 2_000_000],
    'learnable_likelihood_std': [1],
    'reset_bnn': [0, 1],
    'sim_prior': ['none_FVSGD', 'none_SVGD', 'high_fidelity', 'low_fidelity', 'high_fidelity_no_aditive_GP', ],
    'include_aleatoric_noise': [1],
    'best_bnn_model': [1],
    'best_policy': [1],
    'predict_difference': [1],
    'margin_factor': [20.0],
    'ctrl_cost_weight': [0.005],
    'num_stacked_actions': [3],
    'delay': [3 / 30],
    'max_replay_size_true_data_buffer': [10_000],
    'likelihood_exponent': [0.5],
    'data_batch_size': [32],
    'bandwidth_svgd': [0.2],
    'length_scale_aditive_sim_gp': [10.0],
    'num_f_samples': [512],
    'num_measurement_points': [16],
}


def main():
    command_list = []

    all_flags_combinations = dict_permutations(applicable_configs)
    for flags in all_flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
