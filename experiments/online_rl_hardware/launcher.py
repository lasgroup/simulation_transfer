import online_rl_loop
from experiments.util import generate_run_commands, generate_base_command, dict_permutations


def main(args):
    _applicable_configs = {
        'prior': ['none_FVSGD', 'high_fidelity', 'low_fidelity',
                  'low_fidelity_grey_box'],
        'seed': list(range(5)),
        'machine': ['local'],
        'gpu': [1],
        'project_name': ['OnlineRLTestFull'],
        'reset_bnn': [1],
        'deterministic_policy': [1],
        'initial_state_fraction': [0.5],
        'bnn_train_steps': [40_000],
        'sac_num_env_steps': [500_000],
        'num_sac_envs': [128],
        'num_env_steps': [100],
        'num_f_samples': [512]
    }

    all_flags_combinations = dict_permutations(_applicable_configs)

    command_list = []
    for flags in all_flags_combinations:
        cmd = generate_base_command(online_rl_loop, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode='euler', duration='3:59:00', prompt=True, mem=16000)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Meta-BO run')
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    main(args)
