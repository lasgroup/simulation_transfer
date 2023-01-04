import optax
import haiku as hk
from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import PyTreeDef

from typing import Any, Tuple
from tqdm import tqdm
from sim_transfer.modules.attention_modules import ScoreNetworkAttentionModel
from sim_transfer.sims.simulator_base import FunctionSimulator
from sim_transfer.sims.mset_sampler import MSetSampler

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class ScoreMatchingEstimator:

    def __init__(self,
                 function_sim: FunctionSimulator,
                 mset_sampler: MSetSampler,
                 rng_key: jax.random.PRNGKey,
                 n_fn_samples: int = 20,
                 mset_size: int = 10,

                 # score network attributes
                 attn_num_layers: int = 4,
                 attn_architecture: Any = ScoreNetworkAttentionModel,
                 attn_dim: int = 64,
                 attn_key_size: int = 32,
                 attn_num_heads: int = 8,

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

        assert function_sim.input_size == self.mset_sampler.dim_x
        self.x_dim = self.mset_sampler.dim_x

        # setup optimizer
        scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=transition_steps,
            decay_rate=lr_decay_rate)

        self.optimizer = optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)

        self.param = None
        self.opt_state = None

        # setup score network model
        model_kwargs = {"x_dim": self.x_dim,
                        "hidden_dim": attn_dim,
                        "layers": attn_num_layers,
                        "key_size": attn_key_size,
                        "num_heads": attn_num_heads,

                        "layer_norm": True,
                        "widening_factor": 2,
                        "dropout_rate": 0.0,
                        "layer_norm_axis": {"last": -1, "lasttwo": (-2, -1)}["last"],
                        }
        self.nn = hk.transform(lambda *args: attn_architecture(**model_kwargs)(*args))

    def sample_x_fx(self, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """ Samples a measurement set and corresponding function values.
            Returns them concatenated along axis=-1. """

        rng_key_mset, rng_key_f = jax.random.split(rng_key)
        # 1) sample measurement set
        mset = self.mset_sampler.sample_mset(rng_key_mset, mset_size=self.mset_size)
        # 2) sample corresponding function values
        f_samples = self.function_sim.sample_function_vals(mset, num_samples=self.n_fn_samples,
                                                           rng_key=rng_key_f)
        # 3) tile measurement set and stack x_mset and f samples together
        mset_tiles = jnp.repeat(mset[None, :, :], self.n_fn_samples, axis=0)
        x_fx = jnp.concatenate([mset_tiles, f_samples], axis=-1)
        assert x_fx.shape == (self.n_fn_samples, self.mset_size, self.x_dim + 1)
        return x_fx

    def step(self) -> float:
        loss, self.param, self.opt_state = self._step(next(self.rng_gen), self.param, self.opt_state)
        return float(loss)

    def train(self, n_iter: int = 20000):
        assert n_iter > 0

        # init if necessary
        self._init_nn_and_optim()

        # training loop
        pbar = tqdm(range(n_iter))
        for _ in pbar:
            loss = self.step()
            pbar.set_description("Training SNN., loss is %s" % loss)

        return float(loss)

    def _aux_nn_apply(self, params_: PyTreeDef, rng_key: jax.random.PRNGKey, x_: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ auxillary forward pass through the nn returning the gradients and outputs simultaneously. """
        ret = self.nn.apply(params_, rng_key, x_, True)
        return ret, ret

    def _loss(self, params: PyTreeDef, x_fx_samples: jnp.array, rng_key: jax.random.PRNGKey) -> jnp.array:
        """ Score matching loss by Hyvarinen. """
        jacobian, score_pred = jax.vmap(jax.jacrev(lambda x_: self._aux_nn_apply(params, rng_key, x_),
                                                   has_aux=True))(x_fx_samples)
        jacobian = jacobian[..., self.x_dim:].squeeze()  # only take part of Jacobian w.r.t. f
        loss = jax.vmap(jnp.trace)(jacobian) + 0.5 * jnp.linalg.norm(score_pred, axis=-1) ** 2
        return loss.mean()

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, rng_key: jax.random.PRNGKey, param: optax.Params,
              opt_state: Any) -> Tuple[float, optax.Params, Any]:
        """ Performs one gradient step on the score matching loss """
        rng_key_data, rng_key_loss = jax.random.split(rng_key)
        x_fx = self.sample_x_fx(rng_key_data)  # sample mset and corresponding function vals
        loss, grads = jax.value_and_grad(self._loss)(param, x_fx, rng_key)  # get score matching loss grads
        updates, opt_state = self.optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)
        return loss, param, opt_state

    def _init_nn_and_optim(self):
        if self.param is None and self.opt_state is None:
            x_fx_init = self.sample_x_fx(next(self.rng_gen))
            self.param = self.nn.init(next(self.rng_gen), x_fx_init[0, ...])
            self.opt_state = self.optimizer.init(self.param)

    def __call__(self, x_fx: jnp.ndarray) -> jnp.ndarray:
        return self.nn.apply(self.param, next(self.rng_gen), x_fx)


if __name__ == '__main__':
    from sim_transfer.sims.simulator_base import GaussianProcessSim
    from sim_transfer.sims.mset_sampler import UniformMSetSampler
    key = jax.random.PRNGKey(9645)
    function_sim = GaussianProcessSim(input_size=1)
    mset_sampler = UniformMSetSampler(l_bound=-5 * jnp.ones(1),
                                      u_bound=-5 * jnp.ones(1))

    est = ScoreMatchingEstimator(function_sim=function_sim,
                                 mset_sampler=mset_sampler,
                                 rng_key=key,
                                 learning_rate=0.001)

    est.train(n_iter=20000)
