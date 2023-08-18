import unittest
import pytest
import jax
import jax.numpy as jnp
from typing import Dict

from sim_transfer.sims.mset_sampler import UniformMSetSampler
from sim_transfer.sims.simulators import GaussianProcessSim, AdditiveSim, FunctionSimulator
from sim_transfer.sims.util import decode_angles, encode_angles, angle_diff

class TestUniformMSetSampler(unittest.TestCase):

    def test_shape_and_range(self):
        mset_size = 25
        dim_x = 4
        l_bound = -3 * jnp.ones(dim_x)
        u_bound = 10 * jnp.ones(dim_x)
        mset_sampler = UniformMSetSampler(l_bound=l_bound,
                                          u_bound=u_bound)
        key = jax.random.PRNGKey(2234)
        mset = mset_sampler.sample_mset(rng_key=key, mset_size=mset_size)
        assert mset.shape == (mset_size, dim_x)
        assert jnp.all((l_bound < mset ) & (mset < u_bound))


class TestGaussianProcessSim(unittest.TestCase):

    def test_shape_1(self):
        # test whether the generated samples have the correct shape
        key = jax.random.PRNGKey(2234)
        dim_x = 4
        function_sim = GaussianProcessSim(input_size=dim_x)
        mset = jax.random.uniform(key, shape=(10, dim_x))
        f_vals = function_sim.sample_function_vals(mset, num_samples=7, rng_key=key)
        assert f_vals.shape == (7, 10, 1)

    def test_shape_2(self):
        # test whether the generated samples have the correct shape
        key = jax.random.PRNGKey(2234)
        dim_x, dim_y = 3, 2
        function_sim = GaussianProcessSim(input_size=dim_x, output_size=dim_y)
        mset = jax.random.uniform(key, shape=(9, dim_x))
        f_vals = function_sim.sample_function_vals(mset, num_samples=12, rng_key=key)
        assert f_vals.shape == (12, 9, dim_y)

    def test_gp_marginals(self):
        """ Test whether the marginal distributions of the GP are correct. """
        sim = GaussianProcessSim(input_size=1, output_size=2,
                                 length_scale=jnp.array([10., 0.01]),
                                 output_scale=jnp.array([10., 0.1]))
        x = jnp.array([[0.], [1.]])
        gp_dist1, gp_dist2 = sim.gp_marginal_dists(x)
        assert gp_dist1.scale_tril[0][0] >= 10.
        assert gp_dist2.scale_tril[0][0] <= 0.11
        cov1 = gp_dist1.scale_tril @ gp_dist1.scale_tril.T
        cov2 = gp_dist2.scale_tril @ gp_dist2.scale_tril.T
        assert cov1[0][1] >= 90
        assert cov2[0][1] <= 1e-4


class _DummySim(FunctionSimulator):

    def __init__(self, input_size: int, output_size: int,
                 fixed_val: float):
        super().__init__(input_size, output_size)
        self.fixed_val = fixed_val

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        f_samples = self.fixed_val * jnp.ones((num_samples, x.shape[0], self.output_size))
        return f_samples

    @property
    def normalization_stats(self) -> Dict[str, jnp.ndarray]:
        return {
            'x_mean': jnp.zeros(self.input_size),
            'x_std': jnp.ones(self.input_size),
            'y_mean': jnp.ones(self.output_size) * self.fixed_val,
            'y_std': jnp.ones(self.output_size) * 0.5
        }


class TestAdditiveSim(unittest.TestCase):

    def testDefaultBehavior1(self):
        key = jax.random.PRNGKey(3455)
        sim = AdditiveSim([_DummySim(input_size=5, output_size=2, fixed_val=3)])
        f_samples = sim.sample_function_vals(x=jnp.zeros((4, 5)), num_samples=10, rng_key=key)
        assert jnp.array_equal(3 * jnp.ones((10, 4, 2)), f_samples)

    def testAddBehavior1(self):
        key = jax.random.PRNGKey(3455)
        sim = AdditiveSim([_DummySim(input_size=5, output_size=2, fixed_val=3),
                           _DummySim(input_size=5, output_size=2, fixed_val=-1),
                           _DummySim(input_size=5, output_size=2, fixed_val=2)])
        f_samples = sim.sample_function_vals(x=jnp.zeros((4, 5)), num_samples=10, rng_key=key)
        assert jnp.array_equal(4 * jnp.ones((10, 4, 2)), f_samples)

    def test_seed_behavior(self):
        sim = AdditiveSim([_DummySim(input_size=3, output_size=2, fixed_val=3),
                           GaussianProcessSim(input_size=3, output_size=2)])
        key1, key2 = jax.random.split(jax.random.PRNGKey(3455), 2)
        f1 = sim.sample_function_vals(x=jnp.zeros((4, 3)), num_samples=7, rng_key=key1)
        f2 = sim.sample_function_vals(x=jnp.zeros((4, 3)), num_samples=7, rng_key=key1)
        f3 = sim.sample_function_vals(x=jnp.zeros((4, 3)), num_samples=7, rng_key=key2)
        assert jnp.array_equal(f1, f2)
        assert not jnp.array_equal(f1, f3)

    def test_normalization_stats(self):
        sim = AdditiveSim([_DummySim(input_size=2, output_size=1, fixed_val=3),
                           _DummySim(input_size=2, output_size=1, fixed_val=-1)])
        norm_stats = sim.normalization_stats
        assert jnp.array_equal(norm_stats['x_mean'], jnp.zeros(2))
        assert jnp.array_equal(norm_stats['x_std'], jnp.ones(2))
        assert jnp.array_equal(norm_stats['y_mean'], 2 * jnp.ones(1))
        assert jnp.array_equal(norm_stats['y_std'], (2 * 0.5**2)**(0.5) * jnp.ones(1))


class TestAngleEncodingDecoding(unittest.TestCase):

    def test_encoding_decoding_consitency(self):
        x = jnp.array([0.2, 0.2, 0.5, -2.3, -0.5, 2.])
        assert jnp.allclose(x, decode_angles(encode_angles(x, angle_idx=2), angle_idx=2))

    def test_encoding_decoding_shapes(self):
        x = jnp.array([-1.2, 5.2, 2.5])
        x_encoded = encode_angles(x, angle_idx=0)
        x_reconst = decode_angles(x_encoded, angle_idx=0)
        assert x_encoded.shape == (4,)
        assert x_reconst.shape == (3,)

    def test_encoding_decoding_boundaries1(self):
        x = jnp.array([-1.2, 5.2, 2.5, -2.3, -234.5, 2.])
        x_encoded = encode_angles(x, angle_idx=0)
        assert jnp.all(-1 <= x_encoded[:2]) and jnp.all(x_encoded[:2] <= 1)

    def test_encoding_decoding_boundaries2(self):
        x = jnp.array([-5.2, 5.2, 2.5, -2.3, -234.5, 2.])
        x_reconst = decode_angles(encode_angles(x, angle_idx=3), angle_idx=3)
        assert -jnp.pi <= x_reconst[3] <= jnp.pi


class TestAngleDiff(unittest.TestCase):
    def test_angle_dist1(self):
        alpha = jnp.array([jnp.pi - 0.1, -jnp.pi + 0.1, 0.1])
        beta = jnp.array([-jnp.pi + 0.1, jnp.pi - 0.3, -0.3])
        dist = angle_diff(alpha, beta)
        assert jnp.allclose(dist, jnp.array([-0.2, 0.4, 0.4]))

    def test_angle_dist2(self):
        alpha = jnp.array([jnp.pi - 0.1, -jnp.pi + 0.1, 0.1, 0.0])
        beta_diff = jnp.array([0.23, -0.4, 1.4, 2.0])
        beta = (alpha + beta_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
        diff = angle_diff(alpha, beta)
        assert jnp.allclose(diff, - beta_diff)

    def test_angle_dist3(self):
        alpha = jnp.array([0.0])
        beta_diff = jnp.array([jnp.pi + 0.5])
        beta = (alpha + beta_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
        diff = angle_diff(alpha, beta)
        assert jnp.allclose(diff, jnp.pi - 0.5)


if __name__ == '__main__':
    pytest.main()