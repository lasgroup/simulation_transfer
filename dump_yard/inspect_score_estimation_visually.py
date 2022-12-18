import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
from matplotlib import pyplot as plt
from sim_transfer.score_estimation import SSGE
from jax.config import config

config.update('jax_disable_jit', True)

dist = tfp.distributions.MultivariateNormalDiag(loc=jnp.array([1., 1.]), scale_diag=jnp.array([1.0, 2.0]))
#dist = tfp.distributions.StudentT(df=jnp.array([5.0]), loc=jnp.array([0.0]), scale=jnp.array([1.0]))
# dist = tfp.distributions.Independent(dist, reinterpreted_batch_ndims=1)

#true_dlog_q_dx, log_q_x = jax.grad(self.get_log_q_x, has_aux=True)(xs)
#dlog_q_dx = self.score_estimator.estimate_gradients_s_x(xs, samples)

key = jax.random.PRNGKey(654)

LB, UB = -3, 3
# x = jnp.linspace(start=LB, stop=UB, num=200).reshape(-1,1)
x1, x2 = jnp.meshgrid(jnp.linspace(-5,5,20),jnp.linspace(-5,5,20))
x = jnp.stack([x1.flatten(), x2.flatten()], axis=-1)

logprob = lambda x: (dist.log_prob(x).sum(), dist.log_prob(x))
score, logp = jax.grad(logprob, has_aux=True)(x)

x_samples = dist.sample(100, seed=key)
ssge = SSGE(eta=0.1, add_linear_kernel=False, n_eigen_threshold=0.98)
score_estimate = ssge.estimate_gradients_s_x(x, x_samples)

plt.quiver(x1, x2, score[:, 0].reshape(x1.shape), score[:, 1].reshape(x1.shape), label='score', color='blue')
plt.quiver(x1, x2, score_estimate[:, 0].reshape(x1.shape), score_estimate[:, 1].reshape(x1.shape), label='score_estimate',
           color='red')
plt.legend()
plt.show()

# plt.plot(x, logp, label='log_prob')
# plt.plot(x, score, label='score')
# plt.plot(x, score_estimate, label='score_estimate')
# plt.scatter(x_samples, jnp.zeros_like(x_samples))
# plt.legend()
# plt.show()