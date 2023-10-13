from collections import OrderedDict
from typing import List, Optional, Callable, Dict, Union
import time
import jax
import jax.numpy as jnp

from sim_transfer.models.bnn import AbstractFSVGD_BNN
from sim_transfer.sims import FunctionSimulator, Domain
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from sim_transfer.score_estimation.score_network_attn import ScoreMatchingEstimator


class BNN_FSVGD_SN(AbstractFSVGD_BNN):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 domain: Domain,
                 rng_key: jax.random.PRNGKey,
                 function_sim: FunctionSimulator,
                 independent_output_dims: bool = True,
                 num_particles: int = 10,
                 num_f_samples: int = 512,
                 num_measurement_points: int = 8,
                 likelihood_std: Union[float, jnp.array] = 0.2,
                 learn_likelihood_std: bool = False,
                 likelihood_exponent: float = 1.0,
                 bandwidth_svgd: float = 0.2,
                 data_batch_size: int = 8,

                 num_msets_per_step: int = 1,
                 lr_sn: float = 1e-3,
                 use_lr_schedule_sn: bool = False,
                 loss_mode_sn: str = 'mm+sliced_sm',
                 num_iter_sm: int = 500000,
                 loss_change_iter_sm: Optional[int] = 350000,
                 save_path_sn: Optional[str] = None,
                 mm_faction_sn: float = 0.5,

                 num_train_steps: int = 10000,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-3,
                 normalize_data: bool = True,
                 normalize_likelihood_std: bool = False,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        rng_key_bnn, self._rng_key_sn = jax.random.split(rng_key)
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key_bnn,
                         data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                         num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=normalization_stats,
                         normalize_likelihood_std=normalize_likelihood_std,
                         lr=lr, weight_decay=weight_decay, likelihood_std=likelihood_std,
                         learn_likelihood_std=learn_likelihood_std, likelihood_exponent=likelihood_exponent,
                         domain=domain, bandwidth_svgd=bandwidth_svgd)
        self.num_measurement_points = num_measurement_points
        self.independent_output_dims = independent_output_dims

        # check and set function sim
        self.function_sim = function_sim
        assert function_sim.output_size == self.output_size and function_sim.input_size == self.input_size
        self.num_f_samples = num_f_samples

        self.num_msets_per_step = num_msets_per_step
        self.lr_sn = lr_sn
        self.use_lr_schedule_sn = use_lr_schedule_sn
        self.loss_mode_sn = loss_mode_sn
        self.num_iter_sm = num_iter_sm
        self.loss_change_iter_sm = loss_change_iter_sm
        self.save_path_sn = save_path_sn
        self.mm_faction_sn = mm_faction_sn
        self.score_net = None

    def setup_sn_eval(self, sample_ms_f_fn: Callable, sim: FunctionSimulator, batch_size: int = 1):
        key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(575756), 4)
        NUM_EVAL_MSETS = 20
        xms = jnp.stack([sample_ms_f_fn(k, mset_size=self.num_measurement_points + batch_size, num_f_samples=1)[0]
                         for k in jax.random.split(jax.random.PRNGKey(234), NUM_EVAL_MSETS)], axis=0)

        def get_true_score(xm, f_samples):
            return jnp.stack([jax.grad(lambda f: jnp.sum(dist.log_prob(f)))(f_samples[..., i])
                              for i, dist in enumerate(sim.gp_marginal_dists(xm))], axis=-1)

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

        def fisher_divergence_vect(xms, est, f_samples):
            assert xms.shape[0] == f_samples.shape[0]
            return jnp.mean(jnp.stack([fisher_divergence(xms[i], est, f_test_samples[i])
                                       for i in range(NUM_EVAL_MSETS)]))

        f_test_samples = jnp.stack([sim.sample_function_vals(xm, num_samples=5000, rng_key=key2)
                                    for xm in xms], axis=0)
        f_train_samples = jnp.stack([sim.sample_function_vals(xm, num_samples=1000, rng_key=key2)
                                     for xm in xms], axis=0)
        f_div_gp = jnp.mean(jnp.stack([fisher_divergence_gp_approx(xms[i], f_test_samples[i], f_train_samples[i])
                                       for i in range(NUM_EVAL_MSETS)]))

        return xms, f_test_samples, fisher_divergence_vect, f_div_gp

    def fit_score_network(self, x_train: jnp.ndarray, num_iter: int = None, log_period: int = 1000,
                          sim: FunctionSimulator = None, log_to_wandb: bool = False):
        if log_to_wandb:
            import wandb

        batch_size = min(self.data_batch_size, x_train.shape[0])
        num_iter = self.num_iter_sm if num_iter is None else num_iter

        def sample_ms_f_fn(rng_key: jax.random.PRNGKey, mset_size: int, num_f_samples: int):
            rng_batch, rng_key_mset, rng_key_f = jax.random.split(rng_key, 3)
            x_batch = jax.random.choice(rng_batch, x_train, shape=(batch_size,), replace=False)
            x_domain = self._sample_measurement_points(rng_key_mset, num_points=mset_size - batch_size)
            mset = jnp.concatenate([x_batch, x_domain], axis=0)
            f_samples = self._fsim_samples(x=mset, key=rng_key_f, num_samples=num_f_samples)
            return mset, f_samples

        if sim is not None:
            xms, f_test_samples, fisher_divergence_vect, f_div_gp = self.setup_sn_eval(
                sample_ms_f_fn=sample_ms_f_fn, sim=sim, batch_size=batch_size)

        if self.score_net == None:
            self.score_net = ScoreMatchingEstimator(
                input_size=self.input_size,
                output_size=self.output_size,
                sample_ms_f_fn=sample_ms_f_fn,
                mset_size=self.num_measurement_points + batch_size,
                num_msets_per_step=self.num_msets_per_step,
                num_f_samples=self.num_f_samples,
                num_iters=self.num_iter_sm,
                loss_change_iter=self.loss_change_iter_sm,
                rng_key=self._rng_key_sn,
                learning_rate=self.lr_sn,
                use_lr_scheduler=self.use_lr_schedule_sn,
                loss_mode=self.loss_mode_sn,
                mm_faction=self.mm_faction_sn,
            )

        loss = self.score_net.train(num_iter=1)
        if sim is not None:
            f_div = fisher_divergence_vect(xms, self.score_net, f_test_samples)
            print(f'Score Network | Iter 1 | Loss: {loss} |  F-div: {f_div:.2f} | F-div GP: {f_div_gp:.2f} ')
            if log_to_wandb:
                wandb.log({'iter': 1, 'loss': loss, 'f_div': f_div, 'f_div_gp': f_div_gp})
        else:
            print(f'Score Network | Iter 1 | Loss: {loss}')
        t = time.time()
        for i in range(1, (num_iter // log_period)+1):
            loss = self.score_net.train(num_iter=log_period)

            itr = i * log_period

            duration = time.time() - t
            if sim is not None:
                t_eval = time.time()
                f_div = fisher_divergence_vect(xms, self.score_net, f_test_samples)
                duration_eval = time.time() - t_eval
                print(f'Score Network | Iter {itr} | Loss: {loss} | F-div: {f_div:.2f} | F-div GP: {f_div_gp:.2f} | '
                      f'Duration: {duration:.2f} sec | Eval Duration: {duration_eval:.2f} sec')
                if log_to_wandb:
                    wandb.log({'iter': itr, 'loss': loss, 'f_div': f_div, 'f_div_gp': f_div_gp,
                               'duration': duration, 'duration_eval': duration_eval,
                               'duration_per_iter': duration/log_period})
            else:
                print(f'Score Network | Iter {itr} | Loss: {loss} | Duration: {duration} sec')
            t = time.time()

            if itr % 50000 == 0:
                self._save_sn_model()

        remaining_iter = num_iter % log_period
        if remaining_iter > 0:
            loss = self.score_net.train(num_iter=remaining_iter)
        self._save_sn_model()
        return loss

    def _save_sn_model(self):
        if self.save_path_sn is not None:
            self.score_net.save_state(self.save_path_sn)
            print('Saved Score Network Model to:', self.save_path_sn)

    def _load_sn_model(self, path):
        self.score_net.load_state(self.save_path_sn)
        print('Loaded Score Network Model from:', self.save_path_sn)

    # def fit(self, x_train: jnp.ndarray, *args, **kwargs):
    #     self.fit_score_network(x_train)
    #     super().fit(x_train, *args, **kwargs)

    def _neg_log_posterior(self, pred_raw: jnp.ndarray, likelihood_std: jnp.array, x_stacked: jnp.ndarray,
                           y_batch: jnp.ndarray, train_data_till_idx: int,
                           num_train_points: Union[float, int], key: jax.random.PRNGKey):
        nll = - num_train_points * self.likelihood_exponent\
              * self._ll(pred_raw, likelihood_std, y_batch, train_data_till_idx)

        prior_score = self.score_net.pred_score(xm=x_stacked, f_vals=pred_raw)
        neg_log_post = nll - jnp.sum(jnp.mean(pred_raw * jax.lax.stop_gradient(prior_score), axis=-2))

        stats = OrderedDict(train_nll_loss=nll)
        if self.learn_likelihood_std:
            stats['likelihood_std'] = jnp.mean(likelihood_std)
        return neg_log_post, stats

    def _fsim_samples(self, x: jnp.array, key: jax.random.PRNGKey, num_samples: Optional[int] = None) -> jnp.ndarray:
        num_samples = self.num_f_samples if num_samples is None else num_samples
        x_unnormalized = self._unnormalize_data(x)
        f_prior = self.function_sim.sample_function_vals(x=x_unnormalized, num_samples=num_samples, rng_key=key)
        f_prior_normalized = self._normalize_y(f_prior)
        return f_prior_normalized


if __name__ == '__main__':
    import argparse
    import time

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--gpu', type=int, default=None)
    arg_parser.add_argument('--lr_scheduler', action='store_true')
    arg_parser.add_argument('--lr', type=float, default=1e-3)

    args = arg_parser.parse_args()

    if args.gpu == None:
        device_type = 'cpu'
        device_num = 0 if args.gpu is None else args.gpu
    else:
        device_type = 'gpu'
        device_num = args.gpu
    jax.config.update("jax_default_device", jax.devices(device_type)[device_num])
    print('using device:', jax.devices(device_type)[device_num])

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
    score_estimator = 'kde'

    x_train = jax.random.uniform(key=next(key_iter), shape=(num_train_points,),
                                 minval=domain.l, maxval=domain.u).reshape(-1, 1)
    y_train = fun(x_train)

    x_test = jnp.linspace(domain.l, domain.u, 100).reshape(-1, 1)
    y_test = fun(x_test)

    print('\nNUM_DIM_X:', NUM_DIM_X)
    print('NUM_DIM_Y:', NUM_DIM_Y)
    print('SIM_TYPE:', SIM_TYPE)

    print('use_lr_schedule_sn:', args.lr_scheduler)
    print('lr:', args.lr)

    bnn = BNN_FSVGD_SN(NUM_DIM_X, NUM_DIM_Y, domain=domain, rng_key=next(key_iter), function_sim=sim,
                       hidden_layer_sizes=[64, 64, 64], num_train_steps=20000, data_batch_size=1,
                       num_particles=20, num_f_samples=512, num_measurement_points=4,
                       bandwidth_svgd=1., normalization_stats=sim.normalization_stats,
                       likelihood_std=0.05, use_lr_schedule_sn=args.lr_scheduler, lr=args.lr,
                       num_iter_sm=80000)

    bnn.fit_score_network(x_train)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
        if NUM_DIM_X == 1:
            bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'FSVGD SimPrior {score_estimator}, itr {(i + 1) * 2000}',
                        domain_l=domain.l[0], domain_u=domain.u[0])
