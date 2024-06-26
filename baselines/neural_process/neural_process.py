import jax
import optax
import time
import wandb

from collections import OrderedDict
from functools import partial
from matplotlib import pyplot as plt
from typing import Optional, Dict, Union, Callable, Tuple, List
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import numpy as jnp

from .np_models import NPEncoderDet, NPEncoderStoch, NPDecoder
from sim_transfer.modules.metrics import calibration_error_cum, calibration_error_bin
from sim_transfer.modules.util import RngKeyMixin, aggregate_stats
from sim_transfer.sims import Domain, FunctionSimulator



class NeuralProcess(RngKeyMixin):

    def __init__(self, input_size: int, output_size: int,
                 domain: Domain,
                 function_sim: FunctionSimulator,
                 rng_key: jax.random.PRNGKey,
                 num_f_samples: int = 16,
                 num_z_samples: int = 4,
                 num_points_target: int = 16,
                 num_points_context: int = 8,
                 latent_dim: int = 128,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 num_transf_blocks: int = 2,
                 use_cross_attention: bool = True,
                 use_self_attention: bool = True,
                 likelihood_std: Union[float, jnp.array, List[float]] = 0.2,
                 lr: float = 1e-3,
                 lr_end: Optional[float] = None,
                 num_train_steps: int = 50000):
        RngKeyMixin.__init__(self, rng_key)
        self.input_size = input_size
        self.output_size = output_size

        self.domain = domain
        self.function_sim = function_sim

        self.num_f_samples = num_f_samples
        self.num_z_samples = num_z_samples
        self.num_points_target = num_points_target
        self.num_points_context = num_points_context

        if type(likelihood_std) == float:
            self.likelihood_std = jnp.ones(output_size) * likelihood_std
        elif type(likelihood_std) == list:
            self.likelihood_std = jnp.array(likelihood_std)
        else:
            self.likelihood_std = likelihood_std

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_transf_blocks = num_transf_blocks

        """ Setup Encoder and Decoder Models """
        self.encoder_det = NPEncoderDet(latent_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                        num_transf_blocks=num_transf_blocks,
                                        use_cross_attention=use_cross_attention,
                                        use_self_attention=use_self_attention)
        self.encoder_stoch = NPEncoderStoch(latent_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                            use_self_attention=use_self_attention,
                                            num_transf_blocks=num_transf_blocks)
        self.decoder = NPDecoder(hidden_layer_sizes=[hidden_dim] * 3, output_size=output_size)

        """ Setup Optimizer """
        self.params = {
            'encoder_det': self.encoder_det.init(self.rng_key, jnp.ones((1, input_size)), jnp.ones((1, output_size)),
                                                 jnp.ones((1, input_size))),
            'encoder_stoch': self.encoder_stoch.init(self.rng_key, jnp.ones((1, input_size)), jnp.ones((1, output_size))),
            'decoder': self.decoder.init(self.rng_key, jnp.ones((1, input_size + 2 * latent_dim)))
        }
        self.num_train_steps = num_train_steps
        if lr_end is None:
            lr_end = lr
        self.lr_scheduler = optax.linear_schedule(init_value=lr, end_value=lr_end, transition_steps=num_train_steps)
        self.optim = optax.adam(self.lr_scheduler)
        self.opt_state = self.optim.init(self.params)

    def _merge_context_target(self, rc: jnp.array, z_samples: jnp.array, x_target: jnp.array) -> jnp.array:
        num_targets = x_target.shape[0]
        num_z_samples = z_samples.shape[-2]
        x_targets_tiled = jnp.repeat(x_target[:, None, :], num_z_samples, axis=-2)
        rc_tiled = jnp.repeat(rc[:, None, :], num_z_samples, axis=-2)
        z_samples_tiled = jnp.repeat(z_samples[None, :], num_targets, axis=-3)
        decoder_input = jnp.concatenate([x_targets_tiled, rc_tiled, z_samples_tiled], axis=-1)
        return decoder_input

    def __call__(self, x_context, y_context, x_target, rng_key: Optional[jax.random.PRNGKey] = None,
                 num_z_samples: int = 64):
        if rng_key is None:
            rng_key = self.rng_key
        rc = self.encoder_det.apply(self.params['encoder_det'], x_context, y_context, x_target)
        mu, sigma = self.encoder_stoch.apply(self.params['encoder_stoch'], x_context, y_context)
        z_samples = tfd.Normal(mu, sigma).sample(sample_shape=(num_z_samples,), seed=rng_key)

        # tile inputs
        decoder_input = self._merge_context_target(rc, z_samples, x_target)
        preds_mean, preds_std = self.decoder.apply(self.params['decoder'], decoder_input)
        assert preds_mean.shape == preds_std.shape == (x_target.shape[0], num_z_samples, self.output_size)
        return preds_mean, preds_std

    def predict_dist(self, x_context: jnp.array, y_context: jnp.array, x_target: jnp.array,
                     rng_key: Optional[jax.random.PRNGKey] = None,
                     num_z_samples: int = 64) -> tfd.Distribution:
        preds_mean, preds_std = self.__call__(x_context, y_context, x_target, rng_key, num_z_samples)
        pred_dist = self._to_pred_dist(preds_mean, preds_std)
        assert pred_dist.batch_shape == x_target.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        return pred_dist

    def predict(self, x_context: jnp.array, y_context: jnp.array, x_target: jnp.array,
                rng_key: Optional[jax.random.PRNGKey] = None,
                num_z_samples: int = 64) -> Tuple[jnp.array, jnp.array]:
        pred_dist = self.predict_dist(x_context, y_context, x_target, rng_key, num_z_samples)
        return pred_dist.mean(), pred_dist.stddev()

    def eval(self, x_context: jnp.array, y_context: jnp.array, x_target: jnp.array, y_target: jnp.array,
             rng_key: Optional[jax.random.PRNGKey] = None, num_z_samples: int = 16, prefix: str = ''):
        pred_dist = self.predict_dist(x_context, y_context, x_target, rng_key=rng_key, num_z_samples=num_z_samples)

        nll = - jnp.mean(pred_dist.log_prob(y_target))
        rmse = jnp.sqrt(jnp.mean(jnp.sum((pred_dist.mean() - y_target) ** 2, axis=-1)))
        cal_err_cum = calibration_error_cum(pred_dist, y_target)
        cal_err_bin = calibration_error_bin(pred_dist, y_target)
        avg_std = jnp.mean(pred_dist.stddev())

        eval_stats = {'nll': nll, 'rmse': rmse, 'avg_std': avg_std,
                      'cal_err_cum': cal_err_cum, 'cal_err_bin': cal_err_bin}
        # add prefix to stats names
        eval_stats = {f'{prefix}{name}': val for name, val in eval_stats.items()}
        # make sure that all stats are python native floats
        eval_stats = {name: float(val) for name, val in eval_stats.items()}
        return eval_stats

    def _to_pred_dist(self, preds_mean: jnp.array, preds_std: jnp.array) -> tfd.Distribution:
        num_z_samples = preds_mean.shape[1]
        independent_normals = tfd.MultivariateNormalDiag(preds_mean, preds_std)
        mixture_distribution = tfd.Categorical(probs=jnp.ones(num_z_samples) / num_z_samples)
        pred_dist_raw = tfd.MixtureSameFamily(mixture_distribution, independent_normals)
        return pred_dist_raw

    def loss(self, params: Dict, rng_key: jax.random.PRNGKey):
        key_x, key_y, key_noise = jax.random.split(rng_key, 3)

        # sample data
        x = self.domain.sample_uniformly(key_x, sample_shape=(self.num_points_context + self.num_points_target,))
        y = self.function_sim.sample_function_vals(x=x, num_samples=self.num_f_samples, rng_key=key_y)
        y += self.likelihood_std * jax.random.normal(key_noise, shape=y.shape)

        x = jnp.repeat(x[None, :, :], self.num_f_samples, axis=0)
        x_context, x_target = jnp.split(x, [self.num_points_context], axis=-2)
        y_context, y_target = jnp.split(y, [self.num_points_context], axis=-2)

        # encode context and full data
        rc_context = self.encoder_det.apply(params['encoder_det'], x_context, y_context, x_target)
        mu_context, sigma_context = self.encoder_stoch.apply(params['encoder_stoch'], x_context, y_context)
        mu_full, sigma_full = self.encoder_stoch.apply(params['encoder_stoch'], x, y)

        z_dist_context = tfd.Normal(mu_context, sigma_context)
        z_dist_full = tfd.Normal(mu_full, sigma_full)
        kl = jnp.mean(jnp.sum(z_dist_context.kl_divergence(z_dist_full), axis=-1))
        z_samples = z_dist_full.sample(sample_shape=(self.num_z_samples,), seed=rng_key)

        # tile and concatenate inputs for decoder
        x_tiled = jnp.repeat(x_target[None, ...], self.num_z_samples, axis=0)
        rc_context_tiled = jnp.repeat(rc_context[None, ...], self.num_z_samples, axis=0)
        z_samples_tilded = jnp.repeat(z_samples[..., None, :], self.num_points_target, axis=-2)
        decoder_input = jnp.concatenate([x_tiled, rc_context_tiled, z_samples_tilded], axis=-1)
        target_pred_mean, target_pred_std = self.decoder.apply(params['decoder'], decoder_input)

        # log-likelihood (sum over y-dim and number of targets, mean over z-samples and f-samples)
        num_targets = x_target.shape[-2]
        ll = jnp.mean(jnp.sum(tfd.Normal(target_pred_mean, target_pred_std).log_prob(y_target), axis=-1))
        loss = - ll + kl / num_targets
        rmse = jnp.sqrt(jnp.mean(jnp.sum((jnp.mean(target_pred_mean, axis=0) - y_target)**2, axis=-1)))
        stats = OrderedDict({'loss': loss, 'kl': kl, 'll': ll, 'rmse': rmse})
        return loss, stats

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, params: Dict, opt_state: optax.OptState, rng_key: jax.random.PRNGKey):
        (loss, stats), grads = jax.value_and_grad(self.loss, has_aux=True)(params, rng_key)
        updates, opt_state = self.optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return stats, params, opt_state

    def step(self):
        stats, self.params, self.opt_state = self._step(self.params, self.opt_state, self.rng_key)
        stats['lr'] = self.lr_scheduler(self.opt_state[1].count)
        return stats

    def meta_fit(self, log_period: int = 5000, num_steps: Optional[int] = None,
                 eval_data: Optional[Tuple[jnp.array, jnp.array, jnp.array, jnp.array]] = None,
                 log_to_wandb: bool = False):
        evaluate = eval_data is not None
        num_steps = self.num_train_steps if num_steps is None else num_steps
        steps_in_period = 0.0
        stats_list = []
        t_start_period = time.time()

        for step in range(1, num_steps+1):
            # perform the train step
            stats = self.step()
            stats_list.append(stats)
            steps_in_period += 1

            if step % log_period == 0 or step == 1:
                duration_sec = time.time() - t_start_period
                duration_per_step = duration_sec / steps_in_period * 1000
                stats_agg = aggregate_stats(stats_list)

                if evaluate:
                    eval_stats = self.eval(*eval_data, prefix='eval_')
                    stats_agg.update(eval_stats)
                if log_to_wandb:
                    wandb.log({f'{n}': float(v) for n, v in stats_agg.items()}, step=step)
                stats_msg = ' | '.join([f'{n}: {v:.4f}' for n, v in stats_agg.items()])
                msg = (f'Step {step}/{num_steps} | {stats_msg} | Duration {duration_sec:.2f} sec | '
                       f'Time per step {duration_per_step:.2f} ms')
                print(msg)

                # reset the attributes we keep track of
                stats_list = []
                samples_cum_period = 0
                t_start_period = time.time()

            if step >= num_steps:
                break

    def plot_1d(self, x_train: jnp.ndarray, y_train: jnp.ndarray,
                domain_l: Optional[float] = None, domain_u: Optional[float] = None,
                true_fun: Optional[Callable] = None, title: Optional[str] = '', show: bool = True):
        assert self.input_size == 1, 'Can only plot if input_size = 1'

        # determine plotting domain
        x_min, x_max = jnp.min(x_train, axis=0), jnp.max(x_train, axis=0)
        width = x_max - x_min
        if domain_l is None:
            domain_l = x_min - 0.3 * width
        if domain_u is None:
            domain_u = x_max + 0.3 * width
        x_plot = jnp.linspace(domain_l, domain_u, 200).reshape((-1, 1))

        # make predictions
        pred_mean, pred_std = self.predict(x_train, y_train, x_plot)

        # draw the plot
        fig, ax = plt.subplots(nrows=1, ncols=self.output_size, figsize=(self.output_size * 4, 4))
        if self.output_size == 1:
            ax = [ax]
        for i in range(self.output_size):
            ax[i].scatter(x_train.flatten(), y_train[:, i], label='train points')
            if true_fun is not None:
                ax[i].plot(x_plot, true_fun(x_plot)[:, i], label='true fun')
            ax[i].plot(x_plot.flatten(), pred_mean[:, i], label='pred mean')
            ax[i].fill_between(x_plot.flatten(), pred_mean[:, i] - pred_std[:, i],
                            pred_mean[:, i] + pred_std[:, i], alpha=0.3)

            if hasattr(self, 'predict_post_samples'):
                y_post_samples = self.predict_post_samples(x_plot)
                for y in y_post_samples:
                    ax[i].plot(x_plot, y[:, i], linewidth=0.2, color='green')

            if title is not None:
                ax[i].set_title(f'Output dimension {i}')
            ax[i].legend()
        fig.suptitle(title)
        if show:
            fig.show()


if __name__ == '__main__':
    from sim_transfer.sims import SinusoidsSim, QuadraticSim, LinearSim

    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key

    key_iter = key_iter()
    NUM_DIM_X = 1
    NUM_DIM_Y = 2
    SIM_TYPE = 'SinusoidsSim'

    if SIM_TYPE == 'QuadraticSim':
        sim = QuadraticSim()
        fun = lambda x: (x - 2) ** 2
    elif SIM_TYPE == 'LinearSim':
        sim = LinearSim()
        fun = lambda x: x
    elif SIM_TYPE == 'SinusoidsSim':
        sim = SinusoidsSim(output_size=NUM_DIM_Y)

        if NUM_DIM_X == 1 and NUM_DIM_Y == 1:
            fun = lambda x: (2 * x + 2 * jnp.sin(2 * x)).reshape(-1, 1)
        elif NUM_DIM_X == 1 and NUM_DIM_Y == 2:
            fun = lambda x: jnp.concatenate([(2 * x + 2 * jnp.sin(2 * x)).reshape(-1, 1),
                                             (- 2 * x + 2 * jnp.cos(1.5 * x)).reshape(-1, 1)], axis=-1)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    domain = sim.domain
    x_measurement = jnp.linspace(domain.l[0], domain.u[0], 50).reshape(-1, 1)

    num_train_points = 3

    x_train = jax.random.uniform(key=next(key_iter), shape=(num_train_points,),
                                 minval=domain.l, maxval=domain.u).reshape(-1, 1)
    y_train = fun(x_train)

    x_test = jnp.linspace(domain.l, domain.u, 100).reshape(-1, 1)
    y_test = fun(x_test)

    model = NeuralProcess(input_size=NUM_DIM_X, output_size=NUM_DIM_Y, rng_key=jax.random.PRNGKey(45642),
                          domain=domain, function_sim=sim, use_self_attention=True)


    for i in range(10):
        model.meta_fit(num_steps=5000, log_period=2000, eval_data=(x_train, y_train, x_test, y_test))
        if NUM_DIM_X == 1:
            model.plot_1d(x_train, y_train, true_fun=fun, title=f'Neural Process, iter {(i + 1) * 5000}',
                        domain_l=domain.l[0], domain_u=domain.u[0])
