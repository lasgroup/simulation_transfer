import jax
import jax.numpy as jnp
import datetime
import argparse
import os
import json
import sys
import time

from typing import Optional
from experiments.util import Logger, hash_dict, NumpyArrayEncoder
from sim_transfer.score_estimation import SSGE, KDE
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax as tfp
from sim_transfer.modules.metrics import avg_cosine_distance


def get_distribution(dist_type: str, num_dim: int):
    if dist_type == 'diagonal_gaussian':
        dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(num_dim), scale_diag=0.1 * jnp.ones(num_dim))
    elif dist_type == 'gp':
        x = jnp.linspace(-2, 2, num=num_dim).reshape((-1, 1))
        mean = ((x / 2) ** 2).reshape((-1,))
        cov = 0.2 * (1e-3 * jnp.eye(num_dim) + tfp.math.psd_kernels.ExponentiatedQuadratic().matrix(x, x))
        dist = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
    elif dist_type == 'student_t_process':
        x = jnp.linspace(-2, 2, num=num_dim).reshape((-1, 1))
        mean_fn = lambda x: jnp.sum((x / 4), axis=-1)
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=0.1, length_scale=0.4)
        dist = tfd.StudentTProcess(df=3, index_points=x, mean_fn=mean_fn, kernel=kernel)
    elif dist_type == 'mixture':
        mix = 0.5
        dist = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=[mix, 1. - mix]),
                                     components_distribution=tfd.MultivariateNormalDiag(loc=[[-1., -1], [1., 1.]],
                                                                                        scale_diag=[[0.5, 0.4],
                                                                                                    [0.5, 0.5]]))
    else:
        raise NotImplementedError
    return dist


def score_matching_exp(
        dist_type: str,
        num_samples: int,
        num_dim: int,
        data_seed: int,
        score_estim: str,
        model_seed: int = 32345,
        bandwidth: Optional[float] = None,
        eta_ssge: float = 0.1,
        add_linear_kernel: bool = False,
        lambda_nu: float = 1e-3,
):
    dist = get_distribution(dist_type=dist_type, num_dim=num_dim)
    key1, key2 = jax.random.split(jax.random.PRNGKey(data_seed))
    model_key = jax.random.PRNGKey(model_seed)

    # density_fn = lambda x: jnp.exp(dist.log_prob(x))
    score_fn = jax.grad(lambda x: jnp.sum(dist.log_prob(x)))

    samples_train = dist.sample(seed=key1, sample_shape=num_samples)
    samples_test = dist.sample(seed=key2, sample_shape=4 * 10 ** 3)
    true_score = score_fn(samples_test)

    t = time.time()
    if score_estim == 'ssge':
        ssge = SSGE(bandwidth=bandwidth, eta=eta_ssge, add_linear_kernel=add_linear_kernel)
        score_estimates = ssge.estimate_gradients_s_x(x_query=samples_test, x_sample=samples_train)
    elif score_estim == 'kde':
        kde = KDE(bandwidth=bandwidth)
        score_estimates = kde.estimate_gradients_s_x(query=samples_test, samples=samples_train)
    elif score_estim == 'nu_method':
        from non_param_estim.kscore.estimators.nu_method import NuMethod
        nu_method = NuMethod(lam=lambda_nu, fixed_bandwidth=bandwidth)
        score_estimates = nu_method.estimate_gradients_s_x(s=samples_train, x=samples_test, rng_key=model_key)
    else:
        raise NotImplementedError(f'Score estimation method {score_estim} not implemented.')
    inference_duration = time.time() - t

    cos_dist = avg_cosine_distance(true_score, score_estimates)
    l2_norms = jnp.linalg.norm(score_estimates - true_score, axis=-1)
    l2_dist = jnp.mean(l2_norms)
    fisher_div = jnp.mean(l2_norms ** 2)

    median_dist_samples = jnp.median(jnp.linalg.norm(samples_train[:, None, :] - samples_train[None, :, :], axis=-1))

    return {'cos_dist': float(cos_dist), 'l2_dist': float(l2_dist), 'fisher_div': float(fisher_div),
            'inference_time_in_sec': float(inference_duration),
            'median_dist_samples': float(median_dist_samples)}

def main(args):
    ''' generate experiment hash and set up redirect of output streams '''

    exp_params = args.__dict__
    exp_result_folder = exp_params.pop('exp_result_folder')
    num_data_seeds = exp_params.pop('num_data_seeds')
    exp_params['add_linear_kernel'] = bool(exp_params['add_linear_kernel'])
    exp_name = exp_params.pop('exp_name')
    exp_hash = hash_dict(exp_params)


    if exp_result_folder is not None:
        os.makedirs(exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(exp_result_folder, f'{exp_hash}.log ')
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    """ Experiment core """
    results_dicts = []
    for data_seed in range(num_data_seeds):
        t_start = time.time()
        eval_metrics = score_matching_exp(**exp_params, data_seed=data_seed)
        t_end = time.time()

        """ Save experiment results and configuration """
        results_dict = {
            'evals': eval_metrics,
            'params': exp_params,
            'duration_total': t_end - t_start
        }
        results_dicts.append(results_dict)

    if exp_result_folder is None:
        from pprint import pprint
        pprint(results_dicts)
    else:
        exp_result_file = os.path.join(exp_result_folder, '%s.json'%exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dicts, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s'%exp_result_file)



if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # general args
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')

    # data parameters
    parser.add_argument('--dist_type', type=str, default='gp')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_dim', type=int, default=6)
    parser.add_argument('--num_data_seeds', type=int, default=1)

    # model parameters
    parser.add_argument('--model_seed', type=int, default=90234)

    parser.add_argument('--score_estim', type=str, default='kde')
    parser.add_argument('--bandwidth', type=float, default=None)
    parser.add_argument('--eta_ssge', type=float, default=0.1)
    parser.add_argument('--add_linear_kernel', type=int, choices=[0, 1], default=0)
    parser.add_argument('--lambda_nu', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
