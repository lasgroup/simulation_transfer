import run_on_gpu_exp
from experiments.util import generate_run_commands, generate_base_command


def main():
    command_list = []
    cmd = generate_base_command(run_on_gpu_exp, flags={})
    command_list.append(cmd)
    generate_run_commands(command_list, num_cpus=1, num_gpus=1, mode='euler', duration='3:59:00', prompt=True, mem=16000)


if __name__ == '__main__':
    main()
