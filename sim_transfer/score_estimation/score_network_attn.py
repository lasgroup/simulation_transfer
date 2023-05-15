import optax
import haiku as hk
import jax
import jax.numpy as jnp
import math
import pickle

from functools import partial
from jax.tree_util import PyTreeDef
from tensorflow_probability.substrates import jax as tfp
from typing import Any, Tuple, Union, Optional, Callable

from sim_transfer.modules.attention_modules import ScoreNetworkAttentionModel
from sim_transfer.sims.simulators import FunctionSimulator
from sim_transfer.sims.mset_sampler import MSetSampler

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class ScoreMatchingEstimator:

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 sample_ms_f_fn: Callable,
                 rng_key: jax.random.PRNGKey,
                 num_f_samples: int = 20,
                 mset_size: int = 5,
                 num_msets_per_step: int = 2,
                 loss_mode: str = 'sm',
                 mm_faction: float = 0.5,
                 num_slices: Optional[int] = None,

                 # score network attributes
                 attn_num_layers: int = 2,
                 attn_dim: int = 32,
                 attn_key_size: int = 16,
                 attn_num_heads: int = 8,
                 activation_fn: Callable = jax.nn.gelu,

                 # optimizer attributes
                 num_iters: int = 100000,
                 loss_change_iter: Optional[int] = None,
                 learning_rate: float = 1e-3,
                 use_lr_scheduler: bool = False,
                 weight_decay: float = 0.,
                 gradient_clipping: Optional[float] = 10.0):
        """
           Args:
               input_size (int): The size of the input to the model.
               output_size (int): The size of the output from the model.
               sample_ms_f_fn (Callable): A function to generate a measurement set and function values.
                       The function should have the following signature:
                            Args:
                                rng_key (jax.random.PRNGKey): Pseudo-Random Number Generator key from JAX library.
                                mset_size (int): The size of the measurement set to be generated.
                                num_f_samples (int): The number of function values to be generated for each
                                            measurement point.
                            Returns:
                                mset: The sampled measurement set.
                                f_samples: The sampled function values corresponding to the measurement set.

               rng_key (jax.random.PRNGKey): Pseudo-Random Number Generator key from JAX library.
                            It is used for all random operations.
               num_f_samples (int): The number of function values to be generated. Default is 20.
               mset_size (int): The size of the measurement set to be generated. Default is 5.
               num_msets_per_step (int): The number of measurement sets generated per step. Default is 2.
               loss_mode (str): The loss function to be used. Default is 'sm'.
                            Options are 'sm', 'sliced_sm', 'score_mm'.
               num_slices (Optional[int]): The number of slices to use if `loss_mode_sn` is True.
                            If None, the number of slices is set to the number of measurement points. Default is None.
               attn_num_layers (int): The number of layers in the attention mechanism of the score network.
               attn_dim (int): The dimension of the attention layer in the score network. Default is 32.
               attn_key_size (int): The size of the keys in the attention mechanism of the score network. Default is 16.
               attn_num_heads (int): The number of attention heads in the score network. Default is 8.
               activation_fn (Callable): The activation function to be used in the score network.
                            Default is the GELU activation function from the JAX library.
               learning_rate (float): The initial learning rate for the optimizer. Default is 1e-3.
               use_lr_scheduler (bool): A flag to indicate whether to use a linear learning rate scheduler.
               weight_decay (float): The weight decay factor for the optimizer. Default is 0.
               gradient_clipping (Optional[float]): The maximum magnitude for gradients.
                            If None, no gradient clipping is performed. Default is 10.0.
           """
        self.input_size = input_size
        self.output_size = output_size
        self.mset_size = mset_size
        self.num_f_samples = num_f_samples
        self.num_msets_per_step = num_msets_per_step
        self.rng_gen = hk.PRNGSequence(rng_key)
        self.num_iters = num_iters

        self.sample_ms_f_fn = sample_ms_f_fn
        assert self._check_sample_ms_f_fn(sample_ms_f_fn)

        assert loss_mode in ['sm', 'sliced_sm', 'mm', 'mm+sliced_sm', 'mm+sm']
        self.loss_mode = loss_mode
        if len(loss_mode.split('+')) == 2:
            self.loss_change_iter = num_iters // 2 if loss_change_iter is None else loss_change_iter
        else:
            self.loss_change_iter = math.inf
        self.mm_faction = mm_faction
        self.num_slices = self.mset_size if num_slices is None else num_slices

        # setup score network
        model_kwargs = {"x_dim": self.input_size,
                        "hidden_dim": attn_dim,
                        "layers": attn_num_layers,
                        "key_size": attn_key_size,
                        "num_heads": attn_num_heads,
                        "output_dim": self.output_size,

                        "layer_norm": True,
                        "widening_factor": 2,
                        "dropout_rate": 0.0,
                        "layer_norm_axis": {"last": -1, "lasttwo": (-2, -1)}["lasttwo"],
                        "fc_layer_activation_fn": activation_fn,
                        }
        self.model = hk.transform(lambda *args: ScoreNetworkAttentionModel(**model_kwargs)(*args))

        # setup optimizer
        if use_lr_scheduler:
            scheduler = optax.linear_schedule(init_value=learning_rate, end_value=1e-4,
                                              transition_steps=self.num_iters,
                                              transition_begin=self.num_iters // 10)
        else:
            scheduler = learning_rate

        if gradient_clipping is None:
            self.optimizer = optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)
        else:
            self.optimizer = optax.chain(optax.clip_by_global_norm(gradient_clipping),
                                         optax.adamw(learning_rate=scheduler, weight_decay=weight_decay))
        self.param = None
        self.opt_state = None

    def sample_x_fx(self, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """ Samples a measurement set and corresponding function values.
            Returns them concatenated along axis=-1. """
        # sample mset and f_samples
        mset, f_samples = self.sample_ms_f_fn(rng_key, self.mset_size, self.num_f_samples)
        # tile mset and concatenate with f_samples
        x_fx = self._combine_xf(mset, f_samples)
        return x_fx

    def train(self, num_iter: Optional[int] = None) -> float:
        assert num_iter > 0, 'must be at least one iteration'

        # init if necessary
        self._init_nn_and_optim()

        num_iter = self.num_iters if num_iter is None else num_iter
        iter_count = 0
        loss_cum = 0.
        # training loop
        for i in range(1, num_iter+1):
            self._itr += 1
            loss, self.param, self.opt_state = self._step_jitted(next(self.rng_gen), self.param, self.opt_state)
            loss = float(loss)
            loss_cum += loss
            iter_count += 1

            if self._itr == self.loss_change_iter:
                self._change_loss()
        return float(loss_cum/iter_count)

    def pred_score(self, xm: jnp.array, f_vals: jnp.array) -> jnp.ndarray:
        x_fx = self._combine_xf(x=xm, f=f_vals)
        return self.__call__(x_fx)

    def save_state(self, path: str):
        assert self.param is not None, 'must train first'
        assert self.opt_state is not None, 'must train first'
        with open(path, 'wb') as f:
            pickle.dump({'params': self.param, 'opt_state': self.opt_state}, f)

    def load_state(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.param = state['params']
        self.opt_state = state['opt_state']

    @partial(jax.jit, static_argnums=(0,))
    def _pred_score(self, params: PyTreeDef, rng_key: jax.random.PRNGKey, xm: jnp.array,
                    f_vals: jnp.array) -> jnp.ndarray:
        x_fx = self._combine_xf(x=xm, f=f_vals)
        return self.model.apply(params, rng_key, x_fx)

    def _change_loss(self):
        second_loss_mode = self.loss_mode.split('+')[1]
        self._step_jitted = jax.jit(partial(self._step, loss_mode=second_loss_mode))
        print(f'Changed loss to {second_loss_mode}')

    def __call__(self, x_fx: jnp.ndarray) -> jnp.ndarray:
        return self.model.apply(self.param, next(self.rng_gen), x_fx)

    def _sm_loss(self, params: PyTreeDef, rng: jax.random.PRNGKey, x_fx: jnp.array) -> jnp.array:
        """ Score matching loss by Hyvarinen. """
        def sn_fwd(x):
            score_pred = self.model.apply(params, rng, x, True)
            return score_pred, score_pred
        jacs, score_preds = jax.vmap(jax.jacrev(sn_fwd, has_aux=True))(x_fx)
        jacs = jacs[..., - self.output_size:]  # only the gradients w.r.t. f_samples are relevant for the sm loss
        trace_loss = jnp.mean(jnp.trace(jnp.trace(jacs, axis1=-4, axis2=-2), axis1=-1, axis2=-2))
        l2_loss = 0.5 * jnp.mean(jnp.sum(score_preds**2, axis=(-2, -1)))
        return trace_loss + l2_loss

    def _sliced_sm_loss_raw(self, params: PyTreeDef, rng: jax.random.PRNGKey, x_fx: jnp.array, v: jnp.array):
        # sliced score matching loss by Song et al. (2019) where the projections v are given as an argument
        def sn_fwd(x, v):
            score_pred = self.model.apply(params, rng, x, True)
            return jnp.sum(score_pred * v), score_pred

        grad_vs, score_pred = jax.grad(sn_fwd, has_aux=True)(x_fx, v)
        loss_1 = 0.5 * jnp.mean(jnp.sum(score_pred ** 2, axis=(-2, -1)))
        loss_2 = jnp.mean(jnp.sum(v * grad_vs[..., -self.output_size:], axis=(-2, -1)))
        return loss_1 + loss_2

    def _sliced_sm_loss(self, params: PyTreeDef, rng: jax.random.PRNGKey, x_fx: jnp.array):
        # sliced score matching loss by Song et al. (2019)
        rng_key_v, rng_key_loss = jax.random.split(rng)
        v = jax.random.normal(rng_key_v, shape=(self.num_slices,) + x_fx.shape[:-1] + (self.output_size,))
        v *= jnp.sqrt(v.shape[-1]) / jnp.linalg.norm(v, axis=-1)[..., None]
        loss = jnp.mean(jax.vmap(self._sliced_sm_loss_raw, in_axes=(None, None, None, 0))(params, rng_key_loss, x_fx, v))
        return loss

    def _score_mm_loss(self, params: PyTreeDef, rng: jax.random.PRNGKey, x_fx: jnp.array) -> jnp.array:
        """ Score matching loss by Hyvarinen. """
        fs = x_fx[..., - self.output_size:]
        mean = jnp.mean(fs, axis=0).T
        cov = jnp.swapaxes(tfp.stats.covariance(fs, sample_axis=0, event_axis=1), 0, -1)
        est_dist = tfd.MultivariateNormalFullCovariance(mean, cov)
        if self.mm_faction < 1.0:
            # only feed a fraction of the samples through the model to save compute
            n_fraction = int(fs.shape[0] * self.mm_faction)
            fs = fs[:n_fraction]
            x_fx = x_fx[:n_fraction]
        score_gp_approx = jax.grad(lambda x: jnp.sum(est_dist.log_prob(jnp.swapaxes(x, axis2=-2, axis1=-1))))(fs)
        score_preds = self.model.apply(params, rng, x_fx, True)
        return jnp.mean(jnp.sum((score_preds - score_gp_approx)**2, axis=(-2, -1)))

    def _sampling_and_loss(self, param: optax.Params, rng_key: jax.random.PRNGKey,
                           itr: Union[int, jnp.array]) -> Union[float, jnp.array]:
        # compute score matching loss
        rng_key_sample, rng_key_loss = jax.random.split(rng_key)
        x_fx = self.sample_x_fx(rng_key_sample)
        if self.loss_mode == 'sliced_sm':
            loss = self._sliced_sm_loss(param, rng_key_loss, x_fx)
        elif self.loss_mode == 'sm':
            loss = self._sm_loss(param, rng_key_loss, x_fx)
        elif self.loss_mode == 'mm':
            loss = self._score_mm_loss(param, rng_key_loss, x_fx)
        elif self.loss_mode == 'mm+sliced_sm':
            loss = jax.lax.cond(itr < self.loss_change_iter, self._score_mm_loss, self._sliced_sm_loss, *(param, rng_key_loss, x_fx))
        else:
            raise NotImplementedError(f'Loss mode {self.loss_mode} not implemented.')
        return loss

    def _step(self, rng_key: jax.random.PRNGKey, param: optax.Params,
              opt_state: Any, loss_mode: str) -> Tuple[float, optax.Params, Any]:
        """ Performs one gradient step on the score matching loss """
        def _sampling_and_loss(param: optax.Params, rng_key: jax.random.PRNGKey):
            rng_key_sample, rng_key_loss = jax.random.split(rng_key)
            x_fx = self.sample_x_fx(rng_key_sample)
            if loss_mode == 'sliced_sm':
                loss = self._sliced_sm_loss(param, rng_key_loss, x_fx)
            elif loss_mode == 'sm':
                loss = self._sm_loss(param, rng_key_loss, x_fx)
            elif loss_mode == 'mm':
                loss = self._score_mm_loss(param, rng_key_loss, x_fx)
            else:
                raise NotImplementedError(f'Loss mode {self.loss_mode} not implemented.')
            return loss

        rng_keys = jax.random.split(rng_key, self.num_msets_per_step)
        _sampling_and_loss_vmap = lambda param, _rng_keys: jnp.mean(
            jax.vmap(_sampling_and_loss, in_axes=(None, 0), out_axes=0)(param, _rng_keys))
        loss, grads = jax.value_and_grad(_sampling_and_loss_vmap)(param, rng_keys)  # get score matching loss grads
        updates, opt_state = self.optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)
        return loss, param, opt_state

    def _init_nn_and_optim(self):
        if self.param is None and self.opt_state is None:
            x_fx_init = self.sample_x_fx(next(self.rng_gen))
            self.param = self.model.init(next(self.rng_gen), x_fx_init)
            self.opt_state = self.optimizer.init(self.param)
            self._itr = 0
            first_loss_mode = self.loss_mode.split('+')[0]
            self._step_jitted = jax.jit(partial(self._step, loss_mode=first_loss_mode))

    def _combine_xf(self, x: jnp.array, f: jnp.array) -> jnp.array:
        # tile measurement set and stack x_mset and f samples together
        assert f.ndim == 3 and x.ndim == 2 and x.shape[-2] == f.shape[-2]
        num_f_samples = f.shape[0]
        mset_tiles = jnp.repeat(x[None, :, :], num_f_samples, axis=0)
        x_fx = jnp.concatenate([mset_tiles, f], axis=-1)
        assert x_fx.shape == (num_f_samples, self.mset_size, self.input_size + self.output_size)
        return x_fx

    def _check_sample_ms_f_fn(self, sample_ms_f_fn: Callable):
        # check if sample_ms_f_fn returns a valid measurement set and f samples
        xm, f_samples = sample_ms_f_fn(next(self.rng_gen), self.mset_size, self.num_f_samples)
        assert xm.shape == (self.mset_size, self.input_size), \
            f'sample_ms_f_fn returns measurement set of shape {xm.shape} instead of {(self.mset_size, self.input_size)}'
        assert f_samples.shape == (self.num_f_samples, self.mset_size, self.output_size), \
            f'sample_ms_f_fn returns f samples of shape {f_samples.shape} instead of ' \
            f'{(self.num_f_samples, self.mset_size, self.output_size)}'
        return True

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


if __name__ == '__main__':
    import argparse
    import time

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--gpu', type=int, default=None)
    args = arg_parser.parse_args()

    if args.gpu == None:
        device_type = 'cpu'
        device_num = 0 if args.gpu is None else args.gpu
    else:
        device_type = 'gpu'
        device_num = args.gpu
    jax.config.update("jax_default_device", jax.devices(device_type)[device_num])
    print('using device:', jax.devices(device_type)[device_num])

    from sim_transfer.sims.simulators import GaussianProcessSim
    from sim_transfer.sims.mset_sampler import UniformMSetSampler

    key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(575756), 4)
    function_sim = GaussianProcessSim(input_size=1, output_size=2, output_scale=jnp.array([1., 1.]),
                                      length_scale=jnp.array([1., 1.]))
    # mset_sampler = UniformMSetSampler(l_bound=-2 * jnp.ones(1),
    #                                   u_bound=-2 * jnp.ones(1))

    NUM_MSETS = 3
    mset_sampler_fixed = DummyMSetSampler(dim_x=1, mset_size=5,
                                          num_msets=NUM_MSETS, key=key1)
    mset_sampler = UniformMSetSampler(l_bound=jnp.array([-2.]),
                                      u_bound=jnp.array([2.]))
    mset_sampler = mset_sampler_fixed
    xms = mset_sampler_fixed._fixed_points

    def get_true_score(xm, f_samples):
        return jnp.stack([jax.grad(lambda f: jnp.sum(dist.log_prob(f)))(f_samples[..., i])
                          for i, dist in enumerate(function_sim.gp_marginal_dists(xm))], axis=-1)

    # get_true_score_vmap = jax.vmap(get_true_score, in_axes=(0, None), out_axes=0)

    def fisher_divergence_gp_approx(xm, samples_eval, samples_train, eps=1e-4):
        mean = jnp.mean(samples_train, axis=0).T
        cov = jnp.swapaxes(tfp.stats.covariance(samples_train, sample_axis=0, event_axis=1), 0, -1)
        score_preds = []
        for i in range(mean.shape[0]):
            gp_approx = tfd.MultivariateNormalFullCovariance(
                loc=mean[i], covariance_matrix=cov[i] + eps * jnp.eye(cov.shape[-1]))
            score_pred = jax.grad(lambda x: jnp.sum(gp_approx.log_prob(x)))(samples_eval[..., i])
            score_preds.append(score_pred)
        score_preds = jnp.stack(score_preds, axis=-1)
        true_scores = get_true_score(xm, samples_eval)
        return jnp.mean(jnp.linalg.norm(score_preds - true_scores, axis=-2) ** 2)

    def fisher_divergence(xm, est, f_samples):
        score_pred = est.pred_score(xm=xm, f_vals=f_samples)
        score = get_true_score(xm, f_samples)
        return jnp.mean(jnp.linalg.norm(score_pred - score, axis=-2) ** 2)

    # def fisher_divergence_ssge(xm, samples_eval, samples_train):
    #     score_pred = SSGE().estimate_gradients_s_x(samples_eval, samples_train)
    #     score = get_true_score(xm, samples_eval)
    #     return jnp.mean(jnp.linalg.norm(score_pred - score, axis=-1) ** 2)

    f_test_samples = jnp.stack([function_sim.sample_function_vals(xm, num_samples=5000, rng_key=key2)
                                for xm in xms], axis=0)
    f_train_samples = jnp.stack([function_sim.sample_function_vals(xm, num_samples=1000, rng_key=key2)
                                for xm in xms], axis=0)
    f_div_gp = jnp.mean(jnp.stack([fisher_divergence_gp_approx(xms[i], f_test_samples[i], f_train_samples[i])
                                   for i in range(NUM_MSETS)]))
    # f_div_test_ssge = jnp.mean(jnp.stack([fisher_divergence_ssge(xms[i], f_test_samples[i], f_train_samples[i])
    #                                for i in range(NUM_MSETS)]))

    LR = 5e-3
    LR_SCHEDULER = False
    LOSS_MODE = 'score_mm'
    print('LR:', LR, 'LR_SCHEDULER:', LR_SCHEDULER, 'LOSS_MODE:', LOSS_MODE)

    def sample_ms_f_fn(rng_key: jax.random.PRNGKey, mset_size: int, num_f_samples: int):
        rng_key_mset, rng_key_f = jax.random.split(rng_key)
        mset = mset_sampler.sample_mset(rng_key_mset, mset_size=mset_size)
        f_samples = function_sim.sample_function_vals(mset, num_samples=num_f_samples, rng_key=rng_key_f)
        return mset, f_samples

    est = ScoreMatchingEstimator(input_size=mset_sampler.dim_x,
                                 output_size=function_sim.output_size,
                                 sample_ms_f_fn=sample_ms_f_fn,
                                 mset_size=5,
                                 num_msets_per_step=2,
                                 num_f_samples=500,
                                 rng_key=key3,
                                 learning_rate=LR,
                                 loss_mode='mm',
                                 activation_fn=jax.nn.gelu,
                                 weight_decay=0.00)

    N_ITER = 2000

    loss = est.train(num_iter=1)
    f_div = jnp.mean(jnp.stack([fisher_divergence(xms[i], est, f_test_samples[i])
                                   for i in range(NUM_MSETS)]))
    print(0, loss, f_div, f_div_gp)
    t = time.time()
    for i in range(1, 200+1):
        loss = est.train(num_iter=N_ITER)
        t_eval = time.time()
        f_div = jnp.mean(jnp.stack([fisher_divergence(xms[i], est, f_test_samples[i])
                                    for i in range(NUM_MSETS)]))
        duration = time.time() - t
        duration_eval = time.time() - t_eval
        print(f'Iter {i*N_ITER} | Loss: {loss:.2f} | F-div {f_div:.2f}  | F-div GP-approx {f_div_gp:.2f} '
              f'| Time: {duration} sec | Time eval: {duration_eval} sec')
        t = time.time()