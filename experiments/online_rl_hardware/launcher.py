import online_rl_loop
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

def main(args):
    _applicable_configs = {
        'prior': ['none_FVSGD', 'none_SVGD', 'high_fidelity', 'low_fidelity'],  # 'high_fidelity_no_aditive_GP'],
        'seed': list(range(5)),
        'run_remote': [0],
        'gpu': [1],
        'wandb_tag': ['gpu' if args.num_gpus > 0 else 'cpu'],
        'project_name': ['OnlineRLDebug3'],
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
