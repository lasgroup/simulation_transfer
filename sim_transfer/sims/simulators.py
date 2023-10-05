from functools import cached_property
from typing import Callable, Tuple, Dict, Optional, List, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap, random
from jax.lax import cond
from tensorflow_probability.substrates import jax as tfp

from sim_transfer.sims.domain import Domain, HypercubeDomain, HypercubeDomainWithAngles
from sim_transfer.sims.dynamics_models import Pendulum, PendulumParams, RaceCar, CarParams
from sim_transfer.sims.util import encode_angles, decode_angles
from sim_transfer.sims.car_sim_config import (DEFAULT_CAR_PARAMS_BICYCLE, DEFAULT_CAR_PARAMS_BLEND,
                                              BOUNDS_CAR_PARAMS_BICYCLE, BOUNDS_CAR_PARAMS_BLEND)


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
                        num_samples_test: int = 10000, obs_noise_std: Union[jnp.ndarray, float] = 0.1,
                        x_support_mode_train: str = 'full', param_mode: str = 'random') \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        key1, key2 = jax.random.split(rng_key, 2)

        # 1) sample x
        x_train, x_test = self._sample_x_data(key1, num_samples_train, num_samples_test,
                                              support_mode_train=x_support_mode_train)
        x = jnp.concatenate([x_train, x_test], axis=0)

        # 2) get function values
        if param_mode == 'typical':
            f = self._typical_f(x)
        elif param_mode == 'random':
            f = self.sample_function_vals(x, num_samples=1, rng_key=key2).squeeze(axis=0)
        else:
            raise ValueError(f'param_mode {param_mode} not supported')

        # 3) add noise
        y = self._add_observation_noise(f_vals=f, obs_noise_std=obs_noise_std, rng_key=key2)

        # 4) split into train and test
        y_train = y[:num_samples_train]
        y_test = y[-num_samples_test:]

        # 5) check shapes and return dataset
        self._check_dataset_shapes(x_train, y_train, x_test, y_test, num_samples_train, num_samples_test)
        return x_train, y_train, x_test, y_test

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    def _sample_x_data(self, rng_key: jax.random.PRNGKey, num_samples_train: int, num_samples_test: int,
                       support_mode_train: str = 'full') -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Sample inputs for training and testing. """
        x_train = self.domain.sample_uniformly(rng_key, num_samples_train, support_mode=support_mode_train)
        x_test = self.domain.sample_uniformly(rng_key, num_samples_test, support_mode='full')
        return x_train, x_test

    def _check_dataset_shapes(self, x_train: jnp.ndarray, y_train: jnp.ndarray,
                              x_test: jnp.ndarray, y_test: jnp.ndarray,
                              num_samples_train: int, num_samples_test: int) -> None:
        # check shapes
        assert x_train.shape == (num_samples_train, self.input_size)
        assert y_train.shape == (num_samples_train, self.output_size)
        assert x_test.shape == (num_samples_test, self.input_size)
        assert y_test.shape == (num_samples_test, self.output_size)

    def _typical_f(self, x: jnp.array) -> jnp.array:
        raise NotImplementedError

    def _add_observation_noise(self, f_vals: jnp.ndarray, obs_noise_std: Union[jnp.ndarray, float],
                               rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        y = f_vals + obs_noise_std * jax.random.normal(rng_key, shape=f_vals.shape)
        assert f_vals.shape == y.shape
        return y


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
        norm_stats = self.base_sims[0].normalization_stats  # take x stats from first sim
        # for the y stats, combine stats from all sims
        for stat_name in ['y_mean', 'y_std']:
            stats_stack = jnp.stack([sim.normalization_stats[stat_name] for sim in self.base_sims], axis=0)
            if 'mean' in stat_name:
                norm_stats[stat_name] = jnp.sum(stats_stack, axis=0)
            else:
                norm_stats[stat_name] = jnp.sqrt(jnp.sum(stats_stack ** 2, axis=0))
        return norm_stats

    def _typical_f(self, x: jnp.array) -> jnp.array:
        raise NotImplementedError


class GaussianProcessSim(FunctionSimulator):

    def __init__(self, input_size: int = 1, output_size: int = 1,
                 output_scale: Union[float, List[float], jnp.array] = 1.0,
                 length_scale: Union[float, List[float], jnp.array] = 1.0,
                 mean_fn: Optional[Callable] = None,
                 consider_only_first_k_dims: Optional[int] = None):
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
            if isinstance(output_scale, list):
                output_scale = jnp.array(output_scale)
            assert output_scale.shape == (output_size,)
            self.output_scales = output_scale

        if isinstance(length_scale, float):
            self.length_scales = length_scale * jnp.ones((output_size,))
        else:
            if isinstance(output_scale, list):
                length_scale = jnp.array(output_scale)
            assert length_scale.shape == (output_size,)
            self.length_scales = length_scale

        # check mean function
        if mean_fn is None:
            # use a zero mean by default
            self.mean_fn = self._zero_mean
        else:
            self.mean_fn = mean_fn

        # self.kernels is a list of kernels, one per output dimension
        self.kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=l) for l in self.length_scales]

        assert consider_only_first_k_dims is None or consider_only_first_k_dims <= input_size
        self.consider_only_first_k_dims = consider_only_first_k_dims

    def _sample_f_val_per_dim(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey,
                              lengthscale: Union[float, jnp.array],
                              output_scale: Union[float, jnp.array]) -> jnp.ndarray:
        if self.consider_only_first_k_dims is not None:
            x = x[..., :self.consider_only_first_k_dims]
        gp_dist = self._gp_marginal_dist(x, lengthscale, output_scale)
        f_samples = gp_dist.sample(sample_shape=(num_samples,), seed=rng_key)
        return f_samples

    def _gp_marginal_dist(self, x: jnp.ndarray, lengthscale: float, output_scale: float, jitter: float = 1e-5) \
            -> tfd.MultivariateNormalFullCovariance:
        """ Returns the marginal distribution of a GP with SE kernel """
        assert x.ndim == 2
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=lengthscale)
        K = kernel.matrix(x, x) + jitter * jnp.eye(x.shape[0])
        m = self.mean_fn(x)
        return tfd.MultivariateNormalFullCovariance(loc=m, covariance_matrix=output_scale ** 2 * K)

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
        f_samples = jax.vmap(self._sample_f_val_per_dim, in_axes=(None, None, 0, 0, 0), out_axes=-1) \
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
        return jnp.repeat(self.mean_fn(x)[:, None], self.output_size, axis=-1)

    @staticmethod
    def _zero_mean(x: jnp.array) -> jnp.array:
        return jnp.zeros((x.shape[0],))

    @cached_property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        key1, key2 = jax.random.split(jax.random.PRNGKey(23423), 2)
        x = self.domain.sample_uniformly(key1, sample_shape=1000)
        norm_stats = {
            'x_mean': jnp.mean(x, axis=0),
            'x_std': jnp.std(x, axis=0),
            'y_mean': jnp.mean(self._typical_f(x), axis=0),
            'y_std': 1.5 * self.output_scales,
        }
        return norm_stats


class EncodeAngleSimWrapper(FunctionSimulator):

    def __init__(self, base_sim: FunctionSimulator, angle_idx: int = 0):
        self.base_sim = base_sim
        self.angle_idx = angle_idx
        super().__init__(input_size=base_sim.input_size, output_size=base_sim.output_size + 1)

    def sample_function_vals(self, *args, **kwargs) -> jnp.ndarray:
        f_samples = self.base_sim.sample_function_vals(*args, **kwargs)
        f_samples = encode_angles(f_samples, angle_idx=self.angle_idx)
        assert f_samples.shape[-1] == self.output_size
        return f_samples

    def _typical_f(self, x: jnp.array) -> jnp.array:
        f = self.base_sim._typical_f(x)
        f = encode_angles(f, angle_idx=self.angle_idx)
        return f

    @cached_property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        norm_stats = self.base_sim.normalization_stats
        norm_stats['y_mean'] = jnp.concatenate(
            [norm_stats['y_mean'][:self.angle_idx],
             jnp.array([0., 0.]),
             norm_stats['y_mean'][self.angle_idx + 1:]])
        norm_stats['y_std'] = jnp.concatenate(
            [norm_stats['y_std'][:self.angle_idx],
             jnp.array([1., 1.]),
             norm_stats['y_std'][self.angle_idx + 1:]])
        assert norm_stats['y_mean'].shape == norm_stats['y_std'].shape == (self.output_size,)
        return norm_stats


class SinusoidsSim(FunctionSimulator):
    amp_mean = 2.0
    amp_std = 0.4
    slope_mean = 2.0
    slope_std = 1.
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


class ShiftedSinusoidsSim(FunctionSimulator):

    def __init__(self):
        super().__init__(input_size=1, output_size=1)

    def _f(self, phase: jnp.ndarray,  x: jnp.ndarray):
        return jnp.sin(2 * jnp.pi * x**2 + phase[:, None, None])

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        phase = jax.random.uniform(rng_key, shape=(num_samples,), minval=-jnp.pi/2, maxval=jnp.pi/2)
        f = self._f(phase, x)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

    def _typical_f(self, x: jnp.array) -> jnp.array:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        f = self._f(jnp.array([0.]), x).reshape(x.shape[0], self.output_size)
        assert f.shape == (x.shape[0], self.output_size)
        return f

    @property
    def domain(self) -> Domain:
        lower = jnp.array([-1.] * self.input_size)
        upper = jnp.array([1.] * self.input_size)
        return HypercubeDomain(lower=lower, upper=upper)

    @cached_property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        norm_stats = {
            'x_mean': jnp.array([0.]),
            'x_std': jnp.array([1.0]),
            'y_mean': jnp.array([0.]),
            'y_std': jnp.array([1.]),
        }
        return norm_stats


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


class LinearBimodalSim(FunctionSimulator):
    def __init__(self):
        super().__init__(input_size=1, output_size=1)
        # Define the intervals.  They should be disjoint.
        self.slope_intervals_neg = jnp.array([-0.6, -0.4])
        self.slope_intervals_pos = jnp.array([[-1.2, -0.8], [0.8, 1.2]])
        # Choose one number uniformly inside the set

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        neg_slopes = jax.random.uniform(rng_key, shape=(num_samples,), minval=self.slope_intervals_neg[0],
                                        maxval=self.slope_intervals_neg[1])

        intervals = jax.random.choice(rng_key, self.slope_intervals_pos, shape=(num_samples,))
        rng_keys = jax.random.split(rng_key, num_samples)

        def one_sample(key, interval):
            return jax.random.uniform(key, shape=(), minval=interval[0], maxval=interval[1])

        pos_slopes = vmap(one_sample, in_axes=(0, 0))(rng_keys, intervals)

        def positive(x, neg_slope, pos_slope):
            return self._f(x, pos_slope)

        def negative(x, neg_slope, pos_slope):
            return self._f(x, neg_slope)

        def fun(x, neg_slope, pos_slope):
            assert x.shape == (self.input_size,) and neg_slope.shape == pos_slope.shape == ()
            return cond(x.reshape() < 0, negative, positive, x, neg_slope, pos_slope)

        fun_multiple_slope = jax.vmap(fun, in_axes=(None, 0, 0), out_axes=0)
        fs = vmap(fun_multiple_slope, in_axes=(0, None, None), out_axes=1)(x, neg_slopes, pos_slopes)
        assert fs.shape == (num_samples, x.shape[0], self.output_size)
        return fs

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

        self._set_default_sampling_bounds()

        # setup domain
        if self.encode_angle:
            self._domain = HypercubeDomainWithAngles(angle_indices=[0], lower=self._domain_lower,
                                                     upper=self._domain_upper)
        else:
            self._domain = HypercubeDomain(lower=self._domain_lower, upper=self._domain_upper)

    def _set_default_sampling_bounds(self):
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
    _default_car_model_params_bicycle: Dict = DEFAULT_CAR_PARAMS_BICYCLE
    _bounds_car_model_params_bicycle: Dict = BOUNDS_CAR_PARAMS_BICYCLE
    _default_car_model_params_blend: Dict = DEFAULT_CAR_PARAMS_BLEND
    _bounds_car_model_params_blend: Dict = BOUNDS_CAR_PARAMS_BLEND

    _dt: float = 1 / 30.
    _angle_idx: int = 2

    # domain for the simulator prior
    _domain_lower = jnp.array([-4., -4., -jnp.pi, -6., -6., -8., -1., -1.])
    _domain_upper = jnp.array([4., 4., jnp.pi, 6., 6., 8., 1., 1.])

    # domain for generating data
    _domain_lower_dataset = jnp.array([-2., -2., -jnp.pi, -2., -2., -3., -1., -1.])
    _domain_upper_dataset = jnp.array([2., 2., jnp.pi, 2., 2., 3., 1., 1.])

    def __init__(self, encode_angle: bool = True, use_blend: bool = False, only_pose: bool = False,
                 no_angular_velocity: bool = False):
        """ Race car simulator

        Args:
            encode_angle: (bool) whether to encode the angle (theta) as sin(theta) and cos(theta)
            use_blend: (bool) whether to use the blend model which captures dynamics or
                        the bicycle model which is only a kinematic model
            only_pose: (bool) whether to predict only the pose (x, y, theta) or also the velocities (vx, vy, vtheta)
        """
        _output_size = (7 if encode_angle else 6) - (3 if only_pose else 0) - (1 if no_angular_velocity else 0)
        FunctionSimulator.__init__(self, input_size=9 if encode_angle else 8,
                                   output_size=_output_size)

        # set up typical parameters
        self.use_blend = use_blend
        _default_params = self._default_car_model_params_blend if use_blend else self._default_car_model_params_bicycle
        self._typical_params = CarParams(**_default_params)

        # modes for different number of outputs
        assert not (only_pose and no_angular_velocity), \
            "Cannot have both only_pose and no_angular_velocity set to True"
        self.only_pose = only_pose
        self.no_angular_velocity = no_angular_velocity

        self.encode_angle = encode_angle
        self.model = RaceCar(dt=self._dt, encode_angle=encode_angle)
        self._state_action_spit_idx = 7 if encode_angle else 6

        # parameter bounds for the 1st part of the dist
        _bounds_car_model_params = self._bounds_car_model_params_blend if use_blend \
            else self._bounds_car_model_params_bicycle
        self._lower_bound_params = CarParams(**{k: jnp.array(v[0]) for k, v in _bounds_car_model_params.items()})
        self._upper_bound_params = CarParams(**{k: jnp.array(v[1]) for k, v in _bounds_car_model_params.items()})
        assert jnp.all(jnp.stack(jtu.tree_flatten(
            jtu.tree_map(lambda l, u: l <= u, self._lower_bound_params, self._upper_bound_params))[0])), \
            'lower bounds have to be smaller than upper bounds'

        # setup domain
        self._domain = self._create_domain(lower=self._domain_lower, upper=self._domain_upper)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        params = self.model.sample_params_uniform(rng_key, sample_shape=(num_samples,),
                                                  lower_bound=self._lower_bound_params,
                                                  upper_bound=self._upper_bound_params)

        def batched_fun(z, params):
            x, u = self._split_state_action(z)
            f = vmap(self.model.next_step, in_axes=(0, 0, None))(x, u, params)
            if self.only_pose:
                f = f[..., :-3]
            elif self.no_angular_velocity:
                f = f[..., :-1]
            return f

        f = vmap(batched_fun, in_axes=(None, 0))(x, params)
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

    def sample_functions(self, num_samples: int, rng_key: jax.random.PRNGKey) -> Callable:
        params = self.model.sample_params_uniform(rng_key, sample_shape=(num_samples,),
                                                  lower_bound=self._lower_bound_params,
                                                  upper_bound=self._upper_bound_params)

        def stacked_fun(z):
            x, u = self._split_state_action(z)
            f = vmap(self.model.next_step, in_axes=(0, 0, 0))(x, u, params)
            if self.only_pose:
                f = f[..., :-3]
            elif self.no_angular_velocity:
                f = f[..., :-1]
            return f

        return stacked_fun

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        if self.encode_angle:
            stats = {'x_mean': jnp.zeros(self.input_size),
                     'x_std': jnp.array([3., 3., 1.0, 1.0, 4.0, 4.0, 5.0, 1.0, 1.0]),
                     'y_mean': jnp.zeros(7),
                     'y_std': jnp.array([3., 3., 1.0, 1.0, 4.0, 4.0, 5.0])}
        else:
            stats = {'x_mean': jnp.zeros(self.input_size),
                     'x_std': jnp.array([3., 3., 2.5, 4.0, 4.0, 5.0, 1.0, 1.0]),
                     'y_mean': jnp.zeros(7),
                     'y_std': jnp.array([3., 3., 2.5, 4.0, 4.0, 5.0])}
        if self.only_pose:
            stats['y_mean'] = stats['y_mean'][:-3]
            stats['y_std'] = stats['y_std'][:-3]
        elif self.no_angular_velocity:
            stats['y_mean'] = stats['y_mean'][:-1]
            stats['y_std'] = stats['y_std'][:-1]
        return stats

    def _typical_f(self, x: jnp.array) -> jnp.array:
        s, u = self._split_state_action(x)
        f = jax.vmap(self.model.next_step, in_axes=(0, 0, None))(s, u, self._typical_params)
        if self.only_pose:
            f = f[..., :-3]
        elif self.no_angular_velocity:
            f = f[..., :-1]
        return f

    def _split_state_action(self, z: jnp.array) -> Tuple[jnp.array, jnp.array]:
        assert z.shape[-1] == self.domain.num_dims
        return z[..., :self._state_action_spit_idx], z[..., self._state_action_spit_idx:]

    def _add_observation_noise(self, f_vals: jnp.ndarray, obs_noise_std: Union[jnp.ndarray, float],
                               rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        if self.only_pose:
            obs_noise_std = obs_noise_std[..., :-3]
        elif self.no_angular_velocity:
            obs_noise_std = obs_noise_std[..., :-1]
        if self.encode_angle:
            # decode angles -> add noise -> encode angles
            f_decoded = decode_angles(f_vals, angle_idx=2)
            y = f_decoded + obs_noise_std * jax.random.normal(rng_key, shape=f_decoded.shape)
            y = encode_angles(y, angle_idx=2)
        else:
            y = f_vals + obs_noise_std * jax.random.normal(rng_key, shape=f_vals.shape)
        assert f_vals.shape == y.shape
        return y

    def _sample_x_data(self, rng_key: jax.random.PRNGKey, num_samples_train: int, num_samples_test: int,
                       support_mode_train: str = 'full') -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Sample inputs for training and testing. """
        dataset_domain = self._create_domain(lower=self._domain_lower_dataset, upper=self._domain_upper_dataset)
        x_train = dataset_domain.sample_uniformly(rng_key, num_samples_train, support_mode=support_mode_train)
        x_test = dataset_domain.sample_uniformly(rng_key, num_samples_test, support_mode='full')
        return x_train, x_test

    def _create_domain(self, lower: jnp.array, upper: jnp.array) -> Domain:
        """ Creates the domain object from the given lower and up bounds. """
        if self.encode_angle:
            return HypercubeDomainWithAngles(angle_indices=[self._angle_idx], lower=lower, upper=upper)
        else:
            return HypercubeDomain(lower=lower, upper=upper)


class PredictStateChangeWrapper(FunctionSimulator):
    def __init__(self, function_simulator: FunctionSimulator):
        """
        Implicitly we assume that the input to the function simulator is of the form z = (x, u) and output is x_next.
        TODO: This is not a good solution. We should prepare more explicit way of doing this.
        """
        self._function_simulator = function_simulator
        input_size = function_simulator.input_size
        output_size = function_simulator.output_size
        self._x_dim = output_size
        self._u_dim = input_size - output_size
        self._z_dim = input_size
        FunctionSimulator.__init__(self, input_size=input_size, output_size=output_size)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.shape[-1] == self._z_dim
        fun_vals = self._function_simulator.sample_function_vals(x=x, num_samples=num_samples, rng_key=rng_key)
        x_delta = fun_vals - x[..., :self._x_dim][None, ...]
        assert x_delta.shape[1:] == x[..., :self._x_dim].shape
        return x_delta

    @property
    def domain(self) -> Domain:
        return self._function_simulator.domain

    def _typical_f(self, x: jnp.array) -> jnp.array:
        x_next = self._function_simulator._typical_f(x)
        return x_next - x[..., :self._x_dim]

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        old_stats = self._function_simulator.normalization_stats
        x = self.domain.sample_uniformly(jax.random.PRNGKey(0), 1000)
        fs = self.sample_function_vals(x, num_samples=10, rng_key=jax.random.PRNGKey(0))
        fs = fs.reshape(-1, self.output_size)

        new_stats = {'x_mean': old_stats['x_mean'],
                     'x_std': old_stats['x_std'],
                     'y_mean': jnp.mean(fs, axis=0),
                     'y_std': 1.5 * jnp.std(fs, axis=0)}
        return new_stats

    def _add_observation_noise(self, *args, **kwargs) -> jnp.ndarray:
        return self._function_simulator._add_observation_noise(*args, **kwargs)

    def _sample_x_data(self, *args, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._function_simulator._sample_x_data(*args, **kwargs)


class StackedActionSimWrapper(FunctionSimulator):

    def __init__(self, function_simulator: FunctionSimulator, num_stacked_actions: int = 3, action_size: int = 2):
        self._function_simulator = function_simulator
        input_size_base = function_simulator.input_size
        output_size_base = function_simulator.output_size
        new_input_size = input_size_base + action_size*num_stacked_actions
        self.action_size = action_size
        self.num_stacked_actions = num_stacked_actions
        self.obs_size = input_size_base - action_size

        FunctionSimulator.__init__(self, input_size=new_input_size, output_size=output_size_base)

    def _expand_vector(self, vector):
        # Expand the vector by repeating the last action_size elements num_stacked_actions times
        return jnp.concatenate([vector[:-self.action_size]] +
                               [vector[-self.action_size:]] * (self.num_stacked_actions + 1))

    @property
    def domain(self) -> Domain:
        base_domain = self._function_simulator.domain
        if isinstance(base_domain, HypercubeDomainWithAngles):
            num_dim_raw = base_domain._lower.shape[0]
            assert all([ind < num_dim_raw - self.action_size for ind in base_domain.angle_indices])
            new_domain = HypercubeDomainWithAngles(
                angle_indices=base_domain.angle_indices,
                lower=self._expand_vector(base_domain._lower),
                upper=self._expand_vector(base_domain._upper)
            )
            assert new_domain.num_dims == self.input_size
            return new_domain
        elif isinstance(base_domain, HypercubeDomain):
            new_domain = HypercubeDomain(
                lower=self._expand_vector(base_domain._lower),
                upper=self._expand_vector(base_domain._upper)
            )
            assert new_domain.num_dims == self.input_size
            return new_domain
        else:
            raise NotImplementedError('StackedActionSimWrapper can currently only handle '
                                      'HypercubeDomain and HypercubeDomainWithAngles domains')

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        base_stats = self._function_simulator.normalization_stats
        base_stats['x_mean'] = self._expand_vector(base_stats['x_mean'])
        base_stats['x_std'] = self._expand_vector(base_stats['x_std'])
        assert base_stats['x_mean'].shape == base_stats['x_std'].shape == (self.input_size,)
        return base_stats

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        x = x[..., :(self.obs_size + self.action_size)]  # take only the "oldest" action
        fun_vals = self._function_simulator.sample_function_vals(x=x, num_samples=num_samples, rng_key=rng_key)
        return fun_vals

    def _add_observation_noise(self, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError


if __name__ == '__main__':
    key1, key2 = jax.random.split(jax.random.PRNGKey(435345), 2)
    function_sim = RaceCarSim(use_blend=False, no_angular_velocity=True)
    x, _ = function_sim._sample_x_data(key1, 1000, 1000)

    f1 = function_sim.sample_function_vals(x, num_samples=10, rng_key=key2)
    f2 = function_sim._typical_f(x)
    print(jnp.isnan(f1).any())
    print(jnp.isnan(f2).any())

    # function_sim.normalization_stats
    # xs = function_sim.domain.sample_uniformly(key, 100)
    # num_f_samples = 20
    # f_vals = function_sim.sample_function_vals(xs, num_samples=num_f_samples, rng_key=key)
    #
    # NUM_PARALLEL = 20
    # fun_stacked = function_sim.sample_functions(num_samples=NUM_PARALLEL, rng_key=key)
    # # fun_stacked = jax.vmap(function_sim._typical_f)
    # fun_stacked = jax.jit(fun_stacked)
    #
    # s = jnp.repeat(jnp.array([-9.5005625e-01, -1.4144412e+00, 9.9892426e-01, 4.6371352e-02,
    #                           7.2260178e-04, 8.1058703e-03, -7.7542849e-03])[None, :], NUM_PARALLEL, axis=0)
    # traj = [s]
    # actions = []
    # for i in range(60):
    #     t = i / 30.
    #     a = jnp.array([- 1 * jnp.cos(2 * t), 0.8 / (t + 1)])
    #     a = jnp.repeat(a[None, :], NUM_PARALLEL, axis=0)
    #     x = jnp.concatenate([s, a], axis=-1)
    #     s = fun_stacked(x)
    #     traj.append(s)
    #     actions.append(a)
    #
    # traj = jnp.stack(traj, axis=0)
    # actions = jnp.stack(actions, axis=0)
    # from matplotlib import pyplot as plt
    #
    # for i in range(NUM_PARALLEL):
    #     plt.plot(traj[:, i, 0], traj[:, i, 1])
    # plt.xlim(-3, 1)
    # plt.ylim(-2, 3)
    # plt.show()
