import optax
import haiku as hk
from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import PyTreeDef

from typing import Any, Tuple, Union
from tqdm import tqdm
from sim_transfer.modules.nn_modules import MLP
from sim_transfer.sims.simulators import FunctionSimulator
from sim_transfer.sims.mset_sampler import MSetSampler

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class ScoreMatchingEstimator:

    def __init__(self,
                 function_sim: FunctionSimulator,
                 mset_sampler: MSetSampler,
                 rng_key: jax.random.PRNGKey,
                 n_fn_samples: int = 20,
                 mset_size: int = 5,
                 num_msets_per_step: int = 2,

                 # optimizer attributes
                 learning_rate: float = 1e-3,
                 transition_steps: int = 500,
                 lr_decay_rate: float = 1.0,
                 weight_decay: float = 0.):

        self.function_sim = function_sim
        self.mset_sampler = mset_sampler
        self.rng_gen = hk.PRNGSequence(rng_key)
        self.n_fn_samples = n_fn_samples
        self.mset_size = mset_size
        self.num_msets_per_step = num_msets_per_step

        assert function_sim.input_size == self.mset_sampler.dim_x
        self.x_dim = self.mset_sampler.dim_x

        # setup score MLP
        self._input_dim_mlp = mset_size * mset_sampler.dim_x + mset_size
        self.model = MLP(input_size=self._input_dim_mlp,
                         output_size=(mset_size * mset_sampler.dim_x),
                         hidden_layer_sizes=(256, 256, 256), hidden_activation=jax.nn.swish)

        # setup optimizer
        scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=transition_steps,
            decay_rate=lr_decay_rate)
        clipping_value = 10.
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(clipping_value),
            optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)
        )
        self.param = None
        self.opt_state = None


    def sample_x_fx(self, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """ Samples a measurement set and corresponding function values.
            Returns them concatenated along axis=-1. """
        rng_key_mset, rng_key_f = jax.random.split(rng_key)
        # sample measurement set
        mset = self.mset_sampler.sample_mset(rng_key_mset, mset_size=self.mset_size)
        # sample function values corresponding to the measurment set with the sim
        f_samples = self.function_sim.sample_function_vals(mset, num_samples=self.n_fn_samples,
                                                           rng_key=rng_key_f)
        # tile mset and concatenate with f_samples
        x_fx = self._combine_xf(mset, f_samples)
        return x_fx

    def step(self) -> float:
        loss, self.param, self.opt_state = self._step(next(self.rng_gen), self.param, self.opt_state)
        return float(loss)

    def train(self, n_iter: int = 20000):
        assert n_iter > 0

        # init if necessary
        self._init_nn_and_optim()

        # training loop
        for i in range(n_iter):
            loss = self.step()
        return float(loss)

    def _loss(self, params: PyTreeDef, x_fx: jnp.array) -> jnp.array:
        """ Score matching loss by Hyvarinen. """
        def sn_fwd(x):
            score_pred = self.model.apply(params, x)
            return score_pred, score_pred
        jacs, score_preds = jax.vmap((jax.jacrev(sn_fwd, has_aux=True)))(x_fx)
        jacs = jacs[..., -self.mset_size:] # only the gradients w.r.t. f_samples are relevant for the sm loss
        loss = jnp.mean(jnp.trace(jacs, axis1=-2, axis2=-1) + 0.5 * jnp.linalg.norm(score_preds, axis=-1) ** 2)
        return loss

    def _sampling_and_loss(self, param: optax.Params, rng_key: jax.random.PRNGKey) -> Union[float, jnp.array]:
        # compute score matching loss
        x_fx = self.sample_x_fx(rng_key)
        loss = self._loss(param, x_fx)
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, rng_key: jax.random.PRNGKey, param: optax.Params,
              opt_state: Any) -> Tuple[float, optax.Params, Any]:
        """ Performs one gradient step on the score matching loss """
        rng_keys = jax.random.split(rng_key, self.num_msets_per_step)
        _sampling_and_loss_vmap = lambda param, rng_key: jnp.mean(
            jax.vmap(self._sampling_and_loss, in_axes=(None, 0), out_axes=0)(param, rng_keys))
        loss, grads = jax.value_and_grad(_sampling_and_loss_vmap)(param, rng_keys)  # get score matching loss grads
        updates, opt_state = self.optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)
        return loss, param, opt_state

    def _init_nn_and_optim(self):
        if self.param is None and self.opt_state is None:
            self.param = self.model.init(next(self.rng_gen), jnp.zeros(self._input_dim_mlp))
            self.opt_state = self.optimizer.init(self.param)

    def _combine_xf(self, x: jnp.array, f: jnp.array) -> jnp.array:
        # tile measurement set and stack x_mset and f samples together
        assert f.ndim == 3 and x.ndim == 2 and x.shape[0] == f.shape[1]
        num_f_samples = f.shape[0]
        mset_tiles = jnp.repeat(x[None, :, :], num_f_samples, axis=0)
        x_fx = jnp.concatenate([mset_tiles, f], axis=-1)
        assert x_fx.shape == (num_f_samples, self.mset_size, self.x_dim + 1)

        # reshape such that x_fx is two dimensional and entries corresponding to f are last in axis -1
        x_fx = jnp.swapaxes(x_fx, -1, -2).reshape((num_f_samples, -1))
        return x_fx

    def pred_score(self, xm: jnp.array, f_vals: jnp.array) -> jnp.ndarray:
        x_fx = self._combine_xf(x=xm, f=f_vals)
        score_preds = self.__call__(x_fx)
        return score_preds

    def __call__(self, x_fx: jnp.ndarray) -> jnp.ndarray:
        return self.model.apply(self.param, x_fx)


if __name__ == '__main__':
    with jax.default_device
    from jax import device_put
    print(device_put(1, jax.devices()[1]).device_buffer.device())

    import time
    class DummyMSetSampler(MSetSampler):

        def __init__(self, dim_x: 1, mset_size: int,
                     key: jax.random.PRNGKey, num_msets: int = 1):
            self.mset_size = mset_size
            self._dim_x = dim_x
            self.key = key
            self.num_msets = num_msets

            self._fixed_points = jax.random.uniform(key, shape=(num_msets, mset_size, self.dim_x),
                                                    minval=-2, maxval=2)

        def sample_mset(self, rng_key: jax.random.PRNGKey, mset_size: int) -> jnp.ndarray:
            assert mset_size == self.mset_size
            idx = jax.random.choice(rng_key, self.num_msets)
            return self._fixed_points[idx]

        @property
        def dim_x(self) -> int:
            return self._dim_x


    from sim_transfer.sims.simulators import GaussianProcessSim
    from sim_transfer.score_estimation import SSGE
    from sim_transfer.sims.mset_sampler import UniformMSetSampler

    key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(575756), 4)
    function_sim = GaussianProcessSim(input_size=1)
    # mset_sampler = UniformMSetSampler(l_bound=-2 * jnp.ones(1),
    #                                   u_bound=-2 * jnp.ones(1))

    NUM_MSETS = 3
    mset_sampler_fixed = DummyMSetSampler(dim_x=1, mset_size=5,
                                          num_msets=NUM_MSETS, key=key1)
    mset_sampler = UniformMSetSampler(l_bound=jnp.array([-2.]),
                                      u_bound=jnp.array([2.]))
    # mset_sampler = mset_sampler_fixed
    xms = mset_sampler_fixed._fixed_points

    def get_true_score(xm, f_samples):
        dist = function_sim.gp_marginal_dist(xm)
        return jax.grad(lambda f: jnp.sum(dist.log_prob(f)))(f_samples)
    #
    # get_true_score_vmap = jax.vmap(get_true_score, in_axes=(0, None), out_axes=0)

    def fisher_divergence_gp_approx(xm, samples_eval, samples_train, eps=1e-4):
        mean = jnp.mean(samples_train, axis=0).T
        cov = jnp.swapaxes(tfp.stats.covariance(samples_train, sample_axis=0, event_axis=1), 0, -1)
        gp_approx = tfd.MultivariateNormalFullCovariance(
            loc=mean, covariance_matrix=cov + eps * jnp.eye(cov.shape[0]))
        score_pred = jax.grad(lambda x: jnp.sum(gp_approx.log_prob(x)))(samples_eval)
        score = get_true_score(xm, samples_eval)
        return jnp.mean(jnp.linalg.norm(score_pred - score, axis=-1) ** 2)

    def fisher_divergence(xm, est, f_samples):
        score_pred = est.pred_score(xm=xm, f_vals=jnp.expand_dims(f_samples, axis=-1))
        score = get_true_score(xm, f_samples)
        return jnp.mean(jnp.linalg.norm(score_pred - score, axis=-1) ** 2)

    def fisher_divergence_ssge(xm, samples_eval, samples_train):
        score_pred = SSGE().estimate_gradients_s_x(samples_eval, samples_train)
        score = get_true_score(xm, samples_eval)
        return jnp.mean(jnp.linalg.norm(score_pred - score, axis=-1) ** 2)

    f_test_samples = jnp.stack([dist.sample(seed=key2, sample_shape=(5000,))
                                for dist in map(function_sim.gp_marginal_dist, xms)], axis=0)
    f_train_samples = jnp.stack([dist.sample(seed=key4, sample_shape=(1000,))
                                 for dist in map(function_sim.gp_marginal_dist, xms)], axis=0)
    f_div_gp = jnp.mean(jnp.stack([fisher_divergence_gp_approx(xms[i], f_test_samples[i], f_train_samples[i])
                                   for i in range(NUM_MSETS)]))
    f_div_test_ssge = jnp.mean(jnp.stack([fisher_divergence_ssge(xms[i], f_test_samples[i], f_train_samples[i])
                                   for i in range(NUM_MSETS)]))


    est = ScoreMatchingEstimator(function_sim=function_sim,
                                 mset_size=5,
                                 num_msets_per_step=2,
                                 n_fn_samples=1000,
                                 mset_sampler=mset_sampler,
                                 rng_key=key3,
                                 learning_rate=0.001,
                                 weight_decay=0.00)

    N_ITER = 2000

    loss = est.train(n_iter=1)
    f_div = jnp.mean(jnp.stack([fisher_divergence(xms[i], est, f_test_samples[i])
                                   for i in range(NUM_MSETS)]))
    print(0, loss, f_div, f_div_gp, f_div_test_ssge)
    t = time.time()
    for i in range(1, 200+1):
        loss = est.train(n_iter=N_ITER)
        t_eval = time.time()
        f_div = jnp.mean(jnp.stack([fisher_divergence(xms[i], est, f_test_samples[i])
                                    for i in range(NUM_MSETS)]))
        duration = time.time() - t
        duration_eval = time.time() - t_eval
        print(f'Iter {i*N_ITER} | Loss: {loss:.2f} | F-div {f_div:.2f}  | F-div GP-approx {f_div_gp:.2f} '
              f'| F-div SSGE {f_div_test_ssge:.2f} | Time: {duration} sec | Time eval: {duration_eval} sec')
        t = time.time()