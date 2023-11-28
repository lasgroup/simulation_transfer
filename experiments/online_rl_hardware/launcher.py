import online_rl_loop
from experiments.util import generate_run_commands, generate_base_command, dict_permutations


def main(args):
    _applicable_configs = {
        'prior': ['none_FVSGD', 'none_SVGD', 'high_fidelity', 'low_fidelity'],
        'seed': list(range(5)),
        'machine': ['local'],
        'gpu': [1],
        'project_name': ['OnlineRLDebug4'],
        'reset_bnn': [1],
        'deterministic_policy': [1],
        'initial_state_fraction': [0.5],
        'bnn_train_steps': [40_000],
        'sac_num_env_steps': [500_000],
        'num_sac_envs': [64],
        'num_env_steps': [200],
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
    parser.add_argument('--num_cpus', type=int, default=2)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    main(args)
