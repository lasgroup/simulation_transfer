import jax
import copy
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any
import brax
from brax.base import System
from brax.envs.inverted_pendulum import InvertedPendulum

from sim_transfer.sims.simulators import FunctionSimulator
from sim_transfer.sims.domain import Domain


def _brax_write_sys(sys, attr, val):
    """Replaces attributes in sys with val."""
    if not attr:
        return sys
    if len(attr) == 2 and attr[0] == 'geoms':
        geoms = copy.deepcopy(sys.geoms)
        if not hasattr(val, '__iter__'):
            for i, g in enumerate(geoms):
                if not hasattr(g, attr[1]):
                    continue
                geoms[i] = g.replace(**{attr[1]: val})
        else:
            sizes = [g.transform.pos.shape[0] for g in geoms]
            g_idx = 0
            for i, g in enumerate(geoms):
                if not hasattr(g, attr[1]):
                    continue
                size = sizes[i]
                geoms[i] = g.replace(**{attr[1]: val[g_idx:g_idx + size].T})
                g_idx += size
        return sys.replace(geoms=geoms)
    if len(attr) == 1:
        return sys.replace(**{attr[0]: val})
    return sys.replace(**{attr[0]: _brax_write_sys(getattr(sys, attr[0]), attr[1:], val)})


def _brax_set_sys(sys, params: Dict[str, jnp.ndarray]):
  """Sets params in the System."""
  for k in params.keys():
    sys = _brax_write_sys(sys, k.split('.'), params[k])
  return sys


class BraxFunctionSimualtor(FunctionSimulator):

    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size=input_size, output_size=output_size)
        assert isinstance(self, brax.envs.env.PipelineEnv), 'needs to inherit from brax env'

        # vmap over random keys and over state_actions
        self._predict_next_random_vmap = jax.vmap(jax.vmap(self._predict_next_random_sys,
                                                           in_axes=(None, 0), out_axes=0),
                                                  in_axes=(0, None), out_axes=0)

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        assert x.shape[-1] == self.input_size and x.ndim == 2
        keys = jax.random.split(rng_key, num_samples)
        f_samples = self._predict_next_random_vmap(keys, x)
        assert f_samples.shape == (num_samples, x.shape[0], self.output_size)
        return f_samples

    @property
    def state_size(self) -> int:
        return self.output_size

    @property
    def action_size(self) -> int:
        return self.input_size - self.output_size

    def _pipeline_step(self, sys: System, pipeline_state: Any, action: jnp.ndarray):
        """Takes a physics step using the physics pipeline."""
        def f(state, _):
            return (self._pipeline.step(sys, state, action, self._debug), None)
        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

    def _predict_next_random_sys(self, rng_key: jax.random.PRNGKey, state_action: jnp.array) -> jnp.array:
        sys = self._randomize_sys(self.sys, rng_key)
        state, action = state_action[:self.state_size], state_action[self.state_size:]
        return self.predict_next(sys, state, action)

    @staticmethod
    def _randomize_sys(sys: System, rng_key: jax.random.PRNGKey) -> System:
        raise NotImplementedError


class RandomInvertedPendulumEnv(InvertedPendulum, BraxFunctionSimualtor):

    def __init__(self, backend='generalized', **kwargs):
        InvertedPendulum.__init__(self, backend=backend, **kwargs)
        BraxFunctionSimualtor.__init__(self, input_size=4 + 1, output_size=4)

    @property
    def domain(self) -> Domain:
        raise NotImplementedError

    def predict_next(self, sys: System, state: jnp.array, action: jnp.array) -> jnp.array:
        q, qd = state[:2], state[2:]
        pipeline_state = self._pipeline.init(sys, q, qd, self._debug)
        pipeline_state = self._pipeline_step(sys, pipeline_state, action)
        new_state = jnp.concatenate([pipeline_state.q, pipeline_state.qd], axis=-1)
        return new_state

    @staticmethod
    def _randomize_sys(sys: System, rng_key: jax.random.PRNGKey) -> System:
        return _brax_set_sys(sys, {
            'link.inertia.mass': sys.link.inertia.mass + jax.random.uniform(rng_key, shape=(sys.num_links(),)),
            # 'geoms.elasticity': jnp.ones_like(sys.geoms[0].elasticity) * jax.random.uniform(rng_key),
        })


if __name__ == '__main__':

    env = RandomInvertedPendulumEnv(backend='generalized')
    key = jax.random.PRNGKey(345354)

    import time

    sys = env.sys
    sys_new = _brax_set_sys(sys, {
        'geoms.elasticity': jnp.ones_like(sys.geoms[0].elasticity) * 0.5,
        #'link.inertia.mass': sys.link.inertia.mass,
        # 'geoms.elasticity': sys.geoms.elasticity + jax.random.uniform(rng_key, shape=(sys.num_geoms(),)),
    })

    predict_next_random_vmap_jit = jax.jit(env._predict_next_random_vmap)
    sample_function_vals_jit = jax.jit(partial(env.sample_function_vals, num_samples=10))


    x = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    print(sample_function_vals_jit(x=x, rng_key=key).shape)


    t = time.time()
    for i in range(100):
        pred = sample_function_vals_jit(x=x, rng_key=key)
    print(time.time() - t)




