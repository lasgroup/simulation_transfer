import jax
import jax.numpy as jnp


class MSetSampler:

    def sample_mset(self, rng_key: jax.random.PRNGKey, mset_size: int):
        raise NotImplementedError

    @property
    def dim_x(self) -> int:
        raise NotImplementedError


class UniformMSetSampler(MSetSampler):

    def __init__(self, l_bound: jnp.array, u_bound: jnp.array) -> None:
        """ Samples measurement sets uniformly from a hyper-cube

        Args:
            l_bound (jnp.array): lower bounds of the hypercube
            u_bound (jnp.array): upper bounds of the hypercube

        """
        super().__init__()
        assert l_bound.shape == u_bound.shape
        assert l_bound.ndim == 1
        assert jnp.all(l_bound <= u_bound), 'lower bound must be smaller than upper bound'

        self.l_bound = l_bound
        self.u_bound = u_bound

    @property
    def dim_x(self) -> int:
        return self.l_bound.shape[0]

    def sample_mset(self, rng_key: jax.random.PRNGKey, mset_size: int) -> jnp.array:
        """ Samples a random measurement set

        Args:
            rng_key (jax.random.PRNGKey): RNG key
            mset_size (int): size of measurement size

        Returns (jnp.array): Array of size (mset_size, dim_x)
        """
        return jax.random.uniform(rng_key, shape=(mset_size, self.dim_x),
                                  minval=self.l_bound, maxval=self.u_bound)
