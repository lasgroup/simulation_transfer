import jax.numpy as jnp
import jax

from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from sim_transfer.modules.metrics import mmd2


def main():
    mean_fn_prior = lambda x: 0.1 * jnp.sin(x).reshape((-1,))
    # kernel1 = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=1.0)
    kernel2 = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=1.0)
    kernel_prior = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=1.0) # tfp.math.psd_kernels.MaternThreeHalves(length_scale=1.0)
    test_fn = lambda x: 0 * jnp.cos(x).reshape((-1,))
    key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)

    def logprob(n):
        index_points = jnp.linspace(-5, 5, n)[:, None]
        eps = 1e-3
        K1 = kernel_prior.matrix(index_points, index_points) + eps * jnp.eye(index_points.shape[0])
        dist1 = tfd.MultivariateNormalFullCovariance(loc=mean_fn_prior(index_points),
                                                     covariance_matrix=K1)
        K2 = kernel2.matrix(index_points, index_points) + eps * jnp.eye(index_points.shape[0])
        dist2 = tfd.MultivariateNormalFullCovariance(loc=test_fn(index_points),
                                                     covariance_matrix=K2)
        f = dist2.sample(seed=key1, sample_shape=100)
        logp, grads = jax.vmap(jax.value_and_grad(lambda x: dist1.log_prob(x)))(f)
        return logp.mean(), jnp.linalg.norm(grads, axis=-1).mean()

    # make actual evaluations
    num_points = list(range(5, 2000, 50))
    logps = []
    grad_norms = []
    for n in num_points:
        logp, grad_norm = logprob(n)
        logps.append(logp)
        grad_norms.append(grad_norm)


    import matplotlib.pyplot as plt
    num_points = jnp.array(num_points)
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(num_points, logps)
    #ax[0].plot(num_points, 10 * jnp.log(num_points), linestyle='--')
    #ax[0].plot(num_points, kl_est_fn(num_points), linestyle='--')
    ax[1].plot(num_points, grad_norms)
    #ax[1].plot(num_points, 10 * jnp.log(num_points), linestyle='--')
    #ax[1].plot(num_points, kl_est_fn(num_points), linestyle='--')
    plt.show()

if __name__ == '__main__':
    main()