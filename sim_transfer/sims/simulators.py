from typing import Callable, Tuple, Dict, Optional, List, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap, random
from functools import cached_property
from tensorflow_probability.substrates import jax as tfp

from sim_transfer.sims.dynamics_models import Pendulum, PendulumParams, RaceCar, CarParams
from sim_transfer.sims.domain import Domain, HypercubeDomain, HypercubeDomainWithAngles


class FunctionSimulator:

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        raise NotImplementedError

    @property
    def domain(self) -> Domain:
        raise NotImplementedError

    def sample_datasets(self, rng_key: jax.random.PRNGKey, num_samples_train: int,
                        num_samples_test: int = 10000, obs_noise_std: float = 0.1,
                        x_support_mode_train: str = 'full', param_mode: str = 'random') \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        key1, key2 = jax.random.split(rng_key, 2)

        # 1) sample x
        x_train = self.domain.sample_uniformly(key1, num_samples_train, support_mode=x_support_mode_train)
        x_test = self.domain.sample_uniformly(key1, num_samples_test, support_mode='full')
        x = jnp.concatenate([x_train, x_test], axis=0)

        # 2) get function values
        if param_mode == 'typical':
            f = self._typical_f(x)
        elif param_mode == 'random':
            f = self.sample_function_vals(x, num_samples=1, rng_key=key2).squeeze(axis=0)
        else:
            raise ValueError(f'param_mode {param_mode} not supported')

        # 3) add noise
        y = f + obs_noise_std * jax.random.normal(key2, shape=f.shape)

        # 4) split into train and test
        y_train = y[:num_samples_train]
        y_test = y[-num_samples_test:]

        # check shapes and return dataset
        assert x_train.shape == (num_samples_train, self.input_size)
        assert y_train.shape == (num_samples_train, self.output_size)
        assert x_test.shape == (num_samples_test, self.input_size)
        assert y_test.shape == (num_samples_test, self.output_size)
        return x_train, y_train, x_test, y_test

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    def _typical_f(self, x: jnp.array) -> jnp.array:
        raise NotImplementedError


class AdditiveSim(FunctionSimulator):
    """
    Forms an additive combination of multiple sims
    """

    def __init__(self, base_sims: List[FunctionSimulator], take_domain_of_idx: int = 0):
        assert len(base_sims) > 0, 'base sims must be a list of at least one sim'
        assert len({sim.input_size for sim in base_sims}) == 1, 'the base sims must have the same input size'
        assert len({sim.output_size for sim in base_sims}) == 1, 'the base sims must have the same output size'
        super().__init__(input_size=base_sims[0].input_size, output_size=base_sims[0].output_size)

        self.base_sims = base_sims
        assert take_domain_of_idx < len(base_sims), 'take_domain_of_idx must be a valid index of base_sims'
        self.take_domain_of_idx = take_domain_of_idx

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        rng_keys = jax.random.split(rng_key, len(self.base_sims))
        f_samples_per_sim = [sim.sample_function_vals(x=x, num_samples=num_samples, rng_key=key)
                             for sim, key in zip(self.base_sims, rng_keys)]
        f_samples = jnp.sum(jnp.stack(f_samples_per_sim, axis=0), axis=0)
        assert f_samples.shape == (num_samples, x.shape[0], self.output_size)
        return f_samples

    @property
    def domain(self) -> Domain:
        return self.base_sims[self.take_domain_of_idx].domain

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        norm_stats = {}
        for stat_name in ['x_mean', 'x_std', 'y_mean', 'y_std']:
            stats_stack = jnp.stack([sim.normalization_stats[stat_name] for sim in self.base_sims], axis=0)
            if 'mean' in stat_name:
                norm_stats[stat_name] = jnp.sum(stats_stack, axis=0)
            else:
                norm_stats[stat_name] = jnp.sqrt(jnp.sum(stats_stack**2, axis=0))
        return norm_stats

    def _typical_f(self, x: jnp.array) -> jnp.array:
        raise NotImplementedError


class GaussianProcessSim(FunctionSimulator):

    def __init__(self, input_size: int = 1, output_size: int = 1, output_scale: Union[float, jnp.array] = 1.0,
                 length_scale: Union[float, jnp.array] = 1.0,
                 mean_fn: Optional[Callable] = None):
        """ Samples functions from a Gaussian Process (GP) with SE kernel
        Args:
            input_size: dimensionality of the inputs
            output_scale: output_scale of the SE kernel (coincides with the std of the GP prior)
            length_scale: lengthscale of the SE kernel
            mean_fn (optional): mean function of the GP. If None, uses a zero mean.
        """
        super().__init__(input_size=input_size, output_size=output_size)

        # set output and lengthscale
        if isinstance(output_scale, float):
            self.output_scales = output_scale * jnp.ones((output_size,))
        else:
            assert output_scale.shape == (output_size,)
            self.output_scales = output_scale

        if isinstance(length_scale, float):
            self.length_scales = length_scale * jnp.ones((output_size,))
        else:
            assert length_scale.shape == (output_size,)
            self.length_scales = length_scale

        # check mean function
        if mean_fn is None:
            # use a zero mean by default
            self.mean_fn = lambda x: jnp.zeros((x.shape[0],))
        else:
            self.mean_fn = mean_fn

        self.kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=l) for l in self.length_scales]

    def _sample_f_val_per_dim(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey,
                              lengthscale: Union[float, jnp.array], output_scale: Union[float, jnp.array]) -> jnp.ndarray:
        gp_dist = self._gp_marginal_dist(x, lengthscale, output_scale)
        f_samples = gp_dist.sample(sample_shape=(num_samples,), seed=rng_key)
        return f_samples

    def _gp_marginal_dist(self, x: jnp.ndarray, lengthscale: float, output_scale: float, jitter=1e-4) \
            -> tfd.MultivariateNormalFullCovariance:
        """ Returns the marginal distribution of a GP with SE kernel """
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=lengthscale)
        K = kernel.matrix(x, x) + jitter * jnp.eye(x.shape[0])
        m = self.mean_fn(x)
        return tfd.MultivariateNormalFullCovariance(loc=m, covariance_matrix=output_scale**2 * K)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """ Samples functions from a Gaussian Process (GP) with SE kernel
            Args:
                x: index/measurement points of size (n, input_size)
                num_samples: number of samples to draw
                rng_key: random number generator key
        """
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        keys = random.split(rng_key, self.output_size)
        # sample f values per dimension via vmap
        f_samples = jax.vmap(self._sample_f_val_per_dim, in_axes=(None, None, 0, 0, 0), out_axes=-1)\
                                            (x, num_samples, keys, self.length_scales, self.output_scales)
        # check final f_sample shape
        assert f_samples.shape == (num_samples, x.shape[0], self.output_size)
        return f_samples

    def gp_marginal_dists(self, x) -> List[tfd.Distribution]:
        return [self._gp_marginal_dist(x=x, lengthscale=l, output_scale=o)
                for l, o in zip(self.length_scales, self.output_scales)]

    @property
    def domain(self) -> Domain:
        lower = jnp.array([-2.] * self.input_size)
        upper = jnp.array([2.] * self.input_size)
        return HypercubeDomain(lower=lower, upper=upper)

    def _typical_f(self, x: jnp.array) -> jnp.array:
        return self.mean_fn(x)

    @cached_property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        key1, key2 = jax.random.split(jax.random.PRNGKey(23423), 2)
        x = self.domain.sample_uniformly(key1, sample_shape=1000)
        y = self.sample_function_vals(x, num_samples=50, rng_key=key2)
        y = y.reshape((-1, self.output_size))
        norm_stats = {
            'x_mean': jnp.mean(x, axis=0),
            'x_std': jnp.std(x, axis=0),
            'y_mean': jnp.mean(y, axis=0),
            'y_std': 1.5 * jnp.std(y, axis=0),
        }
        return norm_stats


class SinusoidsSim(FunctionSimulator):
    amp_mean = 2.0
    amp_std = 0.4
    slope_mean = 2.0
    slope_std = 0.3
    freq1_mid = 2.0
    freq1_spread = 0.3
    freq2_mid = 1.5
    freq2_spread = 0.2

    def __init__(self, input_size: int = 1, output_size: int = 1):
        assert input_size == 1, 'only 1 dimensional inputs are supported'
        assert output_size in [1, 2], 'only 1 or 2-dimensional outputs are supported'
        super().__init__(input_size=input_size, output_size=output_size)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)
        freq = jax.random.uniform(key1, shape=(num_samples,), minval=self.freq1_mid - self.freq1_spread,
                                  maxval=self.freq1_mid + self.freq1_spread)
        amp = self.amp_mean + self.amp_std * jax.random.normal(key2, shape=(num_samples,))
        slope = self.slope_mean + self.slope_std * jax.random.normal(key3, shape=(num_samples,))
        f = self._f1(amp[:, None, None], freq[:, None, None], slope[:, None, None], x)
        if self.output_size == 2:
            freq2 = jax.random.uniform(key4, shape=(num_samples,), minval=self.freq2_mid - self.freq2_spread,
                                       maxval=self.freq2_mid + self.freq2_spread)
            f2 = self._f2(amp[:, None, None], freq2[:, None, None], slope[:, None, None], x)
            f = jnp.concatenate([f, f2], axis=-1)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

    def _f1(self, amp, freq, slope, x):
        return amp * jnp.sin(freq * x) + slope * x

    def _f2(self, amp, freq, slope, x):
        return amp * jnp.cos(freq * x) - slope * x

    def _typical_f(self, x: jnp.array) -> jnp.array:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        f = self._f1(self.amp_mean, self.freq1_mid, self.slope_mean, x)
        if self.output_size == 2:
            f2 = self._f2(self.amp_mean, self.freq2_mid, self.slope_mean, x)
            f = jnp.concatenate([f, f2], axis=-1)
        assert f.shape == (x.shape[0], self.output_size)
        return f

    @property
    def domain(self) -> Domain:
        lower = jnp.array([-5.] * self.input_size)
        upper = jnp.array([5.] * self.input_size)
        return HypercubeDomain(lower=lower, upper=upper)

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        return {'x_mean': (self.domain.u + self.domain.l) / 2,
                'x_std': (self.domain.u - self.domain.l) / 2,
                'y_mean': jnp.zeros(self.output_size),
                'y_std': 8 * jnp.ones(self.output_size)}


class QuadraticSim(FunctionSimulator):
    def __init__(self):
        super().__init__(input_size=1, output_size=1)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        ks = jax.random.uniform(rng_key, shape=(num_samples,), minval=0.8, maxval=1.2)
        f = ks[:, None, None] * (x - 2) ** 2
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

    @property
    def domain(self) -> Domain:
        lower = jnp.array([0.] * self.input_size)
        upper = jnp.array([4.] * self.input_size)
        return HypercubeDomain(lower=lower, upper=upper)

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        return {'x_mean': (self.domain.u + self.domain.l) / 2,
                'x_std': (self.domain.u - self.domain.l) / 2,
                'y_mean': 2 * jnp.ones(self.output_size),
                'y_std': 1.5 * jnp.ones(self.output_size)}

    def _typical_f(self, x: jnp.array) -> jnp.array:
        assert x.shape[-1] == self.input_size and x.ndim == 2
        f = (x - 2) ** 2
        assert f.shape == (x.shape[0], self.output_size)
        return f


class LinearSim(FunctionSimulator):

    def __init__(self):
        super().__init__(input_size=1, output_size=1)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        slopes = jax.random.uniform(rng_key, shape=(num_samples,), minval=-1, maxval=1.0)
        f = self._f(x, slopes[:, None, None])
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

    def _f(self, x, slope):
        return slope * x

    @property
    def domain(self) -> Domain:
        lower = jnp.array([-2.] * self.input_size)
        upper = jnp.array([2.] * self.input_size)
        return HypercubeDomain(lower=lower, upper=upper)

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        return {'x_mean': (self.domain.u + self.domain.l) / 2,
                'x_std': (self.domain.u - self.domain.l) / 2,
                'y_mean': jnp.zeros(self.output_size),
                'y_std': 1.5 * jnp.ones(self.output_size)}


class PendulumSim(FunctionSimulator):
    _domain_lower = jnp.array([-jnp.pi, -5, -2.5])
    _domain_upper = jnp.array([jnp.pi, 5, 2.5])
    _typical_params_lf = PendulumParams(m=jnp.array(1.), l=jnp.array(1.), g=jnp.array(9.81), nu=jnp.array(0.0),
                                     c_d=jnp.array(0.0))
    _typical_params_hf = PendulumParams(m=jnp.array(1.), l=jnp.array(1.), g=jnp.array(9.81), nu=jnp.array(0.5),
                                        c_d=jnp.array(0.5))

    def __init__(self, dt: float = 0.05,
                 upper_bound: Optional[PendulumParams] = None,
                 lower_bound: Optional[PendulumParams] = None,
                 encode_angle: bool = True,
                 high_fidelity: bool = False):
        super().__init__(input_size=4 if encode_angle else 3, output_size=3 if encode_angle else 2)
        self.model = Pendulum(dt=dt, dt_integration=0.005, encode_angle=encode_angle)

        self.high_fidelity = high_fidelity
        self.encode_angle = encode_angle
        self._state_action_spit_idx = 3 if encode_angle else 2

        # set bounds for uniform sampling
        self._lower_bound_params, self._upper_bound_params = self._default_sampling_bounds()
        if lower_bound is not None:
            self._lower_bound_params = lower_bound
        if upper_bound is not None:
            self._upper_bound_params = upper_bound
        assert jnp.all(jnp.stack(jtu.tree_flatten(jtu.tree_map(
            lambda l, u: l <= u, self._lower_bound_params, self._upper_bound_params))[0])), \
            'lower bounds have to be smaller than upper bounds'


        if self.encode_angle:
            self._domain = HypercubeDomainWithAngles(angle_indices=[0], lower=self._domain_lower,
                                                     upper=self._domain_upper)
        else:
            self._domain = HypercubeDomain(lower=self._domain_lower, upper=self._domain_upper)

    def _split_state_action(self, z: jnp.array) -> Tuple[jnp.array, jnp.array]:
        assert z.shape[-1] == self.domain.num_dims
        return z[..., :self._state_action_spit_idx], z[..., self._state_action_spit_idx:]

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        params = self.model.sample_params_uniform(rng_key, sample_shape=(num_samples,),
                                                  lower_bound=self._lower_bound_params,
                                                  upper_bound=self._upper_bound_params)

        def batched_fun(z, params):
            x, u = self._split_state_action(z)
            return vmap(self.model.next_step, in_axes=(0, 0, None))(x, u, params)

        f = vmap(batched_fun, in_axes=(None, 0))(x, params)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

    def _default_sampling_bounds(self):
        """ Bounds for uniform sampling of the parameters"""
        if self.high_fidelity:
            lower_bound = PendulumParams(m=jnp.array(.5), l=jnp.array(.5), g=jnp.array(5.0), nu=jnp.array(0.4),
                                             c_d=jnp.array(0.4))
            upper_bound = PendulumParams(m=jnp.array(1.5), l=jnp.array(1.5), g=jnp.array(15.0), nu=jnp.array(0.6),
                                             c_d=jnp.array(0.6))
        else:
            lower_bound = PendulumParams(m=jnp.array(.5), l=jnp.array(.5), g=jnp.array(5.0), nu=jnp.array(0.0),
                                             c_d=jnp.array(0.0))
            upper_bound = PendulumParams(m=jnp.array(1.5), l=jnp.array(1.5), g=jnp.array(15.0), nu=jnp.array(0.0),
                                             c_d=jnp.array(0.0))

        return lower_bound, upper_bound

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        if self.encode_angle:
            return {'x_mean': jnp.zeros(self.input_size),
                    'x_std': jnp.array([1., 1., 3.5, 2.]),
                    'y_mean': jnp.zeros(self.output_size),
                    'y_std': jnp.array([1., 1., 3.5])}
        else:
            return {'x_mean': jnp.zeros(self.input_size),
                    'x_std': jnp.array([2.5, 3.5, 2.0]),
                    'y_mean': jnp.zeros(self.output_size),
                    'y_std': jnp.array([2.5, 3.5])}

    def _typical_f(self, x: jnp.array) -> jnp.array:
        s, u = self._split_state_action(x)
        typical_params = self._typical_params_hf if self.high_fidelity else self._typical_params_lf
        return self.model.next_step(x=s, u=u, params=typical_params)


class PendulumBiModalSim(PendulumSim):

    def __init__(self, dt: float = 0.05, encode_angle: bool = True,
                 high_fidelity: bool = False):
        FunctionSimulator.__init__(self, input_size=4 if encode_angle else 3, output_size=3 if encode_angle else 2)
        self.model = Pendulum(dt=dt, dt_integration=0.005, encode_angle=encode_angle)

        self.high_fidelity = high_fidelity
        self.encode_angle = encode_angle
        self._state_action_spit_idx = 3 if encode_angle else 2

        self._set_detault_sampling_bounds()

        # setup domain
        if self.encode_angle:
            self._domain = HypercubeDomainWithAngles(angle_indices=[0], lower=self._domain_lower,
                                                     upper=self._domain_upper)
        else:
            self._domain = HypercubeDomain(lower=self._domain_lower, upper=self._domain_upper)

    def _set_detault_sampling_bounds(self):
        # parameter bounds for the 1st part of the dist
        if self.high_fidelity:
            self._lower_bound_params1 = PendulumParams(
                m=jnp.array(.5), l=jnp.array(.5), g=jnp.array(10.), nu=jnp.array(0.4), c_d=jnp.array(0.4))
            self._upper_bound_params1 = PendulumParams(
                m=jnp.array(0.7), l=jnp.array(0.7), g=jnp.array(10.), nu=jnp.array(0.6), c_d=jnp.array(0.6))
        else:
            self._lower_bound_params1 = PendulumParams(
                m=jnp.array(.5), l=jnp.array(.5), g=jnp.array(10.), nu=jnp.array(0.0), c_d=jnp.array(0.0))
            self._upper_bound_params1 = PendulumParams(
                m=jnp.array(0.7), l=jnp.array(0.7), g=jnp.array(10.), nu=jnp.array(0.0), c_d=jnp.array(0.0))

        assert jnp.all(jnp.stack(jtu.tree_flatten(
            jtu.tree_map(lambda l, u: l <= u, self._lower_bound_params1, self._upper_bound_params1))[0])), \
            'lower bounds have to be smaller than upper bounds'

        # parameter bounds for the 2nd part of the dist
        if self.high_fidelity:
            # parameter bounds for the 2nd part of the dist
            self._lower_bound_params2 = PendulumParams(
                m=jnp.array(1.3), l=jnp.array(1.3), g=jnp.array(10.), nu=jnp.array(0.4), c_d=jnp.array(0.4))
            self._upper_bound_params2 = PendulumParams(
                m=jnp.array(1.5), l=jnp.array(1.5), g=jnp.array(10.), nu=jnp.array(0.6), c_d=jnp.array(0.6))
        else:
            self._lower_bound_params2 = PendulumParams(
                m=jnp.array(1.3), l=jnp.array(1.3), g=jnp.array(10.), nu=jnp.array(0.0), c_d=jnp.array(0.0))
            self._upper_bound_params2 = PendulumParams(
                m=jnp.array(1.5), l=jnp.array(1.5), g=jnp.array(10.), nu=jnp.array(0.0), c_d=jnp.array(0.0))

        assert jnp.all(jnp.stack(jtu.tree_flatten(
            jtu.tree_map(lambda l, u: l <= u, self._lower_bound_params2, self._upper_bound_params2))[0])), \
            'lower bounds have to be smaller than upper bounds'

    def _split_state_action(self, z: jnp.array) -> Tuple[jnp.array, jnp.array]:
        assert z.shape[-1] == self.domain.num_dims
        return z[..., :self._state_action_spit_idx], z[..., self._state_action_spit_idx:]

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size

        # split num_samples samples across the two modes
        num_samples1 = int(num_samples / 2)
        num_samples2 = num_samples - num_samples1
        assert num_samples1 + num_samples2 == num_samples

        # samples parameters from the two modes / distribution parts
        params1 = self.model.sample_params_uniform(rng_key, sample_shape=(num_samples1,),
                                                   lower_bound=self._lower_bound_params1,
                                                   upper_bound=self._upper_bound_params1)

        params2 = self.model.sample_params_uniform(rng_key, sample_shape=(num_samples2,),
                                                   lower_bound=self._lower_bound_params2,
                                                   upper_bound=self._upper_bound_params2)

        # concatenate the samples
        params = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=-1), params1, params2)

        def batched_fun(z, params):
            x, u = self._split_state_action(z)
            return vmap(self.model.next_step, in_axes=(0, 0, None))(x, u, params)

        f = vmap(batched_fun, in_axes=(None, 0))(x, params)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        if self.encode_angle:
            return {'x_mean': jnp.zeros(self.input_size),
                    'x_std': jnp.array([1., 1., 3.5, 2.]),
                    'y_mean': jnp.zeros(self.output_size),
                    'y_std': jnp.array([1., 1., 3.5])}
        else:
            return {'x_mean': jnp.zeros(self.input_size),
                    'x_std': jnp.array([2.5, 3.5, 2.0]),
                    'y_mean': jnp.zeros(self.output_size),
                    'y_std': jnp.array([2.5, 3.5])}

    def _typical_f(self, x: jnp.array) -> jnp.array:
        raise NotImplementedError('Does not make sense for bi-modal simulator')


class RaceCarSim(FunctionSimulator):

    def __init__(self, dt: float = 0.01, encode_angle: bool = True, use_blend: bool = False):
        self.use_blend = use_blend
        self._typical_f = CarParams(use_blend=self.use_blend)
        FunctionSimulator.__init__(self, input_size=9 if encode_angle else 8, output_size=7 if encode_angle else 6)
        self.model = RaceCar(dt=dt, encode_angle=encode_angle)
        self._state_action_spit_idx = 7 if encode_angle else 6
        self.encode_angle = encode_angle
        self.max_steering = 0.35
        # parameter bounds for the 1st part of the dist
        self._lower_bound_params = CarParams(
            m=jnp.array(.04), i_com=jnp.array(1e-6), l_f=jnp.array(0.025), l_r=jnp.array(0.025), g=jnp.array(9.0),
            d_f=jnp.array(0.015), c_f=jnp.array(1.2), b_f=jnp.array(2.2), d_r=jnp.array(0.015), c_r=jnp.array(1.2),
            b_r=jnp.array(2.2), c_m_1=jnp.array(0.2), c_m_2=jnp.array(0.04), c_rr=jnp.array(0.001), c_d=jnp.array(0.01),
            use_blend=jnp.array(0.0), steering_limit=jnp.array(self.max_steering))
        self._upper_bound_params = CarParams(
            m=jnp.array(.08), i_com=jnp.array(5e-6), l_f=jnp.array(0.05), l_r=jnp.array(0.05), g=jnp.array(10.0),
            d_f=jnp.array(0.025), c_f=jnp.array(1.5), b_f=jnp.array(2.5), d_r=jnp.array(0.025), c_r=jnp.array(1.5),
            b_r=jnp.array(2.8), c_m_1=jnp.array(0.4), c_m_2=jnp.array(0.07), c_rr=jnp.array(0.01), c_d=jnp.array(0.1),
            use_blend=jnp.array(self.use_blend), steering_limit=jnp.array(self.max_steering))

        assert jnp.all(jnp.stack(jtu.tree_flatten(
            jtu.tree_map(lambda l, u: l <= u, self._lower_bound_params, self._upper_bound_params))[0])), \
            'lower bounds have to be smaller than upper bounds'
        if self.encode_angle:
            self._domain_lower = jnp.array([-50, -50, -jnp.pi, -10, -10, 10, -self.max_steering, -1])
            self._domain_upper = jnp.array([50, 50, jnp.pi, 10, 10, 10, self.max_steering, 1])
            self._domain = HypercubeDomainWithAngles(angle_indices=[2], lower=self._domain_lower,
                                                     upper=self._domain_upper)
        else:
            self._domain_lower = jnp.array([-50, -50, -jnp.pi, -10, -10, 10, -self.max_steering, -1])
            self._domain_upper = jnp.array([50, 50, jnp.pi, 10, 10, 10, self.max_steering, 1])
            self._domain = HypercubeDomain(lower=self._domain_lower, upper=self._domain_upper)

    def _split_state_action(self, z: jnp.array) -> Tuple[jnp.array, jnp.array]:
        assert z.shape[-1] == self.domain.num_dims
        return z[..., :self._state_action_spit_idx], z[..., self._state_action_spit_idx:]

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        params = self.model.sample_params_uniform(rng_key, sample_shape=(num_samples,),
                                                  lower_bound=self._lower_bound_params,
                                                  upper_bound=self._upper_bound_params)
        def batched_fun(z, params):
            x, u = self._split_state_action(z)
            return vmap(self.model.next_step, in_axes=(0, 0, None))(x, u, params)

        f = vmap(batched_fun, in_axes=(None, 0))(x, params)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        if self.encode_angle:
            return {'x_mean': jnp.zeros(self.input_size),
                    'x_std': jnp.array([5, 5, 1.0, 1.0, 1.5, 1.5, 1.5, 1.0, self.max_steering]),
                    'y_mean': jnp.zeros(self.output_size),
                    'y_std': jnp.array([5, 5, 1.0, 1.0, 1.5, 1.5, 1.5])}
        else:
            return {'x_mean': jnp.zeros(self.input_size),
                    'x_std': jnp.array([5, 5, 2.5, 1.5, 1.5, 1.5, 1.0, self.max_steering]),
                    'y_mean': jnp.zeros(self.output_size),
                    'y_std': jnp.array([5, 5, 2.5, 1.5, 1.5, 1.5])}

    def _typical_f(self, x: jnp.array) -> jnp.array:
        s, u = self._split_state_action(x)
        return self.model.next_step(x=s, u=u, params=self._typical_params)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    key = jax.random.PRNGKey(2234)
    dim_x = 4
    function_sim = GaussianProcessSim(input_size=dim_x)
    mset = jax.random.uniform(key, shape=(10, dim_x))
    f_vals = function_sim.sample_function_vals(mset, num_samples=7, rng_key=key)
    # key = jax.random.PRNGKey(675)
    #sim = GaussianProcessSim(input_size=1, output_scale=3.0, mean_fn=lambda x: jnp.squeeze(2 * x, axis=-1))
    #f_vals = sim.sample_function_vals(x=jax.random.normal(jax.random.PRNGKey(12312), (10, sim.input_size)), num_samples=5, rng_key=key)
    # sim = RaceCarSim()
    # sim.sample_function_vals(x=jnp.zeros((1, sim.input_size)), num_samples=5, rng_key=key)
    # plt, axes = plt.subplots(ncols=1, figsize=(4.5, 4))
    # for i in range(1):
    #     x_plot = jnp.linspace(sim.domain.l, sim.domain.u, 200).reshape((-1, 1))
    #     y_samples = sim.sample_function_vals(x_plot, 10, key)
    #     for y in y_samples:
    #         axes.plot(x_plot, y[:, i])

    # axes[0].set_title('Output dimension 1')
    # axes[1].set_title('Output dimension 2')
    # x, y = sim.sample_dataset(key, 100, obs_noise_std=0.1, x_support_mode='partial', param_mode='random')
    # plt.scatter(x, y)
    # plt.show()
