import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'OfflineRLHW_without_frame_stack'

_applicable_configs = {
    'horizon_len': [200],
    'seed': list(range(5)),
    'project_name': [PROJECT_NAME],
    'sac_num_env_steps': [2_000_000],
    'num_epochs': [50],
    'max_train_steps': [100_000],
    'min_train_steps': [40_000],
    'learnable_likelihood_std': ['yes'],
    'include_aleatoric_noise': [0],
    'best_bnn_model': [1],
    'best_policy': [1],
    'margin_factor': [20.0],
    'ctrl_cost_weight': [0.005],
    'ctrl_diff_weight': [1.0],
    'num_offline_collected_transitions': [20, 50, 100, 200, 400, 800, 1600, 2000, 2500, 10000],
    'test_data_ratio': [0.0],
    'eval_on_all_offline_data': [1],
    'eval_only_on_init_states': [1],
    'share_of_x0s_in_sac_buffer': [0.5],
    'bnn_batch_size': [8],
    'likelihood_exponent': [0.5],
    'train_sac_only_from_init_states': [0],
    'data_from_simulation': [0],
    'num_frame_stack': [3],
    'bandwidth_svgd': [0.1, 0.2],
    'length_scale_aditive_sim_gp': [10.0],
    'input_from_recorded_data': [1],
}

_applicable_configs_no_sim_prior = {'use_sim_prior': [0],
                                    'use_grey_box': [0],
                                    'high_fidelity': [0],
                                    'predict_difference': [1],
                                    'num_measurement_points': [8]
                                    } | _applicable_configs
_applicable_configs_high_fidelity = {'use_sim_prior': [1],
                                     'use_grey_box': [0],
                                     'high_fidelity': [1],
                                     'predict_difference': [1],
                                     'num_measurement_points': [8]} | _applicable_configs
_applicable_configs_low_fidelity = {'use_sim_prior': [1],
                                    'use_grey_box': [0],
                                    'high_fidelity': [0],
                                    'predict_difference': [1],
                                    'num_measurement_points': [8]} | _applicable_configs

_applicable_configs_grey_box = {'use_sim_prior': [0],
                                'high_fidelity': [0],
                                'use_grey_box': [1],
                                'predict_difference': [1],
                                'num_measurement_points': [8]} | _applicable_configs

# all_flags_combinations = dict_permutations(_applicable_configs_no_sim_prior) + dict_permutations(
#     _applicable_configs_high_fidelity) + dict_permutations(_applicable_configs_low_fidelity) + dict_permutations(
#     _applicable_configs_grey_box)

all_flags_combinations = dict_permutations(_applicable_configs_no_sim_prior) + dict_permutations(
    _applicable_configs_high_fidelity) + dict_permutations(_applicable_configs_low_fidelity)  # + dict_permutations(


# _applicable_configs_grey_box)


def main():
    command_list = []
    for flags in all_flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=1, num_gpus=0, mode='euler', duration='3:59:00', prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
