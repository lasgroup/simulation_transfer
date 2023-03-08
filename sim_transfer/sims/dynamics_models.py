from abc import ABC, abstractmethod
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random, vmap
from jaxtyping import PyTree


class PendulumParams(NamedTuple):
    m: jax.Array = jnp.array(0.15)
    l: jax.Array = jnp.array(0.5)
    g: jax.Array = jnp.array(9.81)
    nu: jax.Array = jnp.array(0.1)


class DynamicsModel(ABC):
    def __init__(self, h: float, x_dim: int, u_dim: int, params_example: PyTree):
        self.dt = h
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.params_example = params_example

    def next_step(self, x: jax.Array, u: jax.Array, params: PyTree) -> jax.Array:
        return x + self.dt * self.ode(x, u, params)

    def ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        return self._ode(x, u, params)

    @abstractmethod
    def _ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        pass

    def random_split_like_tree(self, key):
        treedef = jtu.tree_structure(self.params_example)
        keys = jax.random.split(key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def sample_params(self, key, upper_bound, lower_bound):
        keys = self.random_split_like_tree(key)
        return jtu.tree_map(
            lambda key, l_bound, u_bound: jax.random.uniform(key, shape=l_bound.shape, minval=l_bound,
                                                             maxval=u_bound), keys, lower_bound, upper_bound)


class Pendulum(DynamicsModel):
    def __init__(self, h):
        super().__init__(h=h, x_dim=2, u_dim=1, params_example=PendulumParams())

    def _ode(self, x, u, params: PendulumParams):
        # x represents [theta in rad/s, theta_dot in rad/s^2]
        # u represents [torque]
        x0_dot = x[1]
        x1_dot = params.g / params.l * jnp.sin(x[0]) - params.nu / (params.m * params.l ** 2) * x[1] + u[0] / (
                params.m * params.l ** 2)
        return jnp.array([x0_dot, x1_dot])


if __name__ == "__main__":
    pendulum = Pendulum(0.1)
    upper_bound = PendulumParams(m=jnp.array(1.0), l=jnp.array(1.0), g=jnp.array(10.0), nu=jnp.array(1.0))
    lower_bound = PendulumParams(m=jnp.array(0.1), l=jnp.array(0.1), g=jnp.array(9.0), nu=jnp.array(0.1))
    key = jax.random.PRNGKey(0)
    keys = random.split(key, 4)
    params = vmap(pendulum.sample_params, in_axes=(0, None, None))(keys, upper_bound, lower_bound)
