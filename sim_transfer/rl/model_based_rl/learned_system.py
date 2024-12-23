from typing import Tuple, Union

import chex
import jax.numpy as jnp
import jax.random as jr
from distrax import Distribution, Normal
from mbpo.systems.base_systems import System, SystemParams
from mbpo.systems.dynamics.base_dynamics import Dynamics
import tensorflow_probability.substrates.jax.distributions as tfd
from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.sims.car_system import CarReward, CarRewardParams, SystemState
from sim_transfer.modules.distribution import ParticleDistribution


@chex.dataclass
class DynamicsParams:
    key: chex.PRNGKey


class LearnedDynamics(Dynamics[DynamicsParams]):
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 model: BatchedNeuralNetworkModel,
                 include_noise: bool = True,
                 predict_difference: bool = True,
                 use_particle_dist: bool = False,
                 sample_with_eps_std: bool = False,
                 num_frame_stack: int = 0
                 ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.model = model
        self.include_noise = include_noise
        self.predict_difference = predict_difference
        self.num_frame_stack = num_frame_stack
        self._x_dim = x_dim - u_dim * num_frame_stack
        self._u_dim = u_dim
        self._use_particle_dist = use_particle_dist
        self._calibration_alpha = 1.0 if sample_with_eps_std else 0.0

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams,
                   ) -> Tuple[Distribution, DynamicsParams, jnp.ndarray]:
        assert x.shape == (self._x_dim + self._u_dim * self.num_frame_stack,) and u.shape == (self._u_dim,)
        # Create state-action pair
        z = jnp.concatenate([x, u])
        z = z.reshape((1, -1))
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        if self.predict_difference:
            delta_x_dist = self.model.predict_dist(z,
                                                   include_noise=self.include_noise,
                                                   use_particle_dist=self._use_particle_dist,
                                                   calibration_alpha=self._calibration_alpha,
                                                   )
            delta_x = delta_x_dist.sample(seed=key_sample_x_next)
            if self._use_particle_dist:
                if isinstance(delta_x_dist, tfd.TransformedDistribution):
                    base_dist = delta_x_dist.distribution
                else:
                    base_dist = delta_x_dist
                assert isinstance(base_dist, ParticleDistribution)
                epistemic_uncertainty = base_dist.raw_epistemic_std
            else:
                epistemic_uncertainty = jnp.zeros_like(delta_x)
            _x = x[:self._x_dim]
            _x_next = _x + delta_x.reshape((self._x_dim,))
        else:
            x_next_dist = self.model.predict_dist(z, include_noise=self.include_noise,
                                                  use_particle_dist=self._use_particle_dist,
                                                  calibration_alpha=self._calibration_alpha,
                                                  )
            _x_next = x_next_dist.sample(seed=key_sample_x_next)
            _x_next = _x_next.reshape((self._x_dim,))
            if self._use_particle_dist:
                assert isinstance(x_next_dist, ParticleDistribution)
                epistemic_uncertainty = x_next_dist.raw_epistemic_std
            else:
                epistemic_uncertainty = jnp.zeros_like(_x_next)

        if self.num_frame_stack > 0:
            # Update last num_frame_stack actions
            _us = x[self._x_dim:]
            new_us = jnp.concatenate([_us[self._u_dim:], u])
            x_next = jnp.concatenate([_x_next, new_us])
        else:
            x_next = _x_next

        # Concatenate state and last num_frame_stack actions
        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(loc=x_next, scale=jnp.zeros_like(x_next)), new_dynamics_params, epistemic_uncertainty

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        return DynamicsParams(key=key)


class LearnedCarSystem(System[DynamicsParams, CarRewardParams]):
    def __init__(self,
                 model: BatchedNeuralNetworkModel,
                 include_noise: bool,
                 predict_difference: bool,
                 num_frame_stack: int = 0,
                 use_optimism: bool = False,
                 sample_with_eps_std: bool = False,
                 intrinsic_reward_weight: Union[float, chex.Array] = 1.0,
                 **car_reward_kwargs: dict):
        self.num_frame_stack = num_frame_stack
        reward = CarReward(**car_reward_kwargs, num_frame_stack=num_frame_stack)
        dynamics = LearnedDynamics(x_dim=reward.x_dim + self.num_frame_stack * reward.u_dim,
                                   u_dim=reward.u_dim, model=model, include_noise=include_noise,
                                   predict_difference=predict_difference,
                                   num_frame_stack=num_frame_stack,
                                   use_particle_dist=use_optimism,
                                   sample_with_eps_std=sample_with_eps_std
                                   )
        self.use_optimism = use_optimism
        System.__init__(self, dynamics=dynamics, reward=reward)
        self._x_dim = reward.x_dim
        self._u_dim = reward.u_dim
        self.intrinsic_reward_weight = intrinsic_reward_weight

    @staticmethod
    def system_params_vmap_axes(axes: int = 0):
        return SystemParams(dynamics_params=DynamicsParams(key=axes),
                            reward_params=CarRewardParams(_goal=None, key=axes),
                            key=axes)

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[DynamicsParams, CarRewardParams],
             ) -> SystemState:
        assert x.shape == (self._x_dim + self._u_dim * self.num_frame_stack,) and u.shape == (self._u_dim,)
        new_key, key_x_next, key_reward = jr.split(system_params.key, 3)
        x_next_dist, next_dynamics_params, epistemic_uncertainty = self.dynamics.next_state(
            x, u, system_params.dynamics_params)
        x_next = x_next_dist.sample(seed=key_x_next)
        reward_dist, next_reward_params = self.reward(x, u, system_params.reward_params, x_next)
        if self.use_optimism:
            reward = reward_dist.sample(seed=key_reward)
            int_reward = jnp.linalg.norm(epistemic_uncertainty, axis=-1) / self._x_dim
            int_reward = int_reward.reshape(reward.shape)
            reward = reward + self.intrinsic_reward_weight * int_reward
        else:
            reward = reward_dist.sample(seed=key_reward)
        return SystemState(x_next=x_next,
                           reward=reward,
                           system_params=SystemParams(dynamics_params=next_dynamics_params,
                                                      reward_params=next_reward_params,
                                                      key=new_key),
                           )
