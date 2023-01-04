import unittest
import pytest
import jax
import jax.numpy as jnp
from sim_transfer.sims.mset_sampler import UniformMSetSampler
from sim_transfer.sims.simulator_base import GaussianProcessSim

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

    def test_shape(self):
        # test whether the generated samples have the correct shape
        key = jax.random.PRNGKey(2234)
        dim_x = 4
        function_sim = GaussianProcessSim(input_size=dim_x)
        mset = jax.random.uniform(key, shape=(10, dim_x))
        f_vals = function_sim.sample_function_vals(mset, num_samples=7, rng_key=key)
        assert f_vals.shape == (7, 10, 1)



if __name__ == '__main__':
    pytest.main()