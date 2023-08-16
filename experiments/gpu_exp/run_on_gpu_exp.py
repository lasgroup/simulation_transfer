import argparse

import jax
import wandb


def jax_has_gpu():
    wandb.init(
        dir='/cluster/scratch/trevenl',
        project='GPU_exp',
    )
    print('Starting experiment')
    wandb.log({'start': 'Starting experiment'})
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        wandb.log({'info': 'JAX has GPU'})
        print('JAX has GPU')
    except:
        print('JAX does not have GPU')
        wandb.log({'info': 'JAX does not have GPU'})

    print('End of experiment')
    wandb.log({'end': 'End of experiment'})
    wandb.finish()


def main(args):
    jax_has_gpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
