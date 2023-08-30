from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from distrax import Distribution, Normal
from mbpo.systems.base_systems import System, SystemParams
from mbpo.systems.dynamics.base_dynamics import Dynamics

from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.sims.car_system import CarReward, CarRewardParams, SystemState


@chex.dataclass
class DynamicsParams:
    key: chex.PRNGKey


class LearnedDynamics(Dynamics[DynamicsParams]):
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 model: BatchedNeuralNetworkModel,
                 include_noise: bool = True,
                 predict_difference: bool = True
                 ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.model = model
        self.include_noise = include_noise
        self.predict_difference = predict_difference

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        # Create state-action pair
        z = jnp.concatenate([x, u])
        z = z.reshape((1, -1))
        if self.predict_difference:
            delta_x_dist = self.model.predict_dist(z, include_noise=self.include_noise)
            next_key, key_sample_x_next = jr.split(dynamics_params.key)
            delta_x = delta_x_dist.sample(seed=key_sample_x_next)
            x_next = x + delta_x.reshape((self.x_dim,))
        else:
            x_next_dist = self.model.predict_dist(z, include_noise=self.include_noise)
            next_key, key_sample_x_next = jr.split(dynamics_params.key)
            x_next = x_next_dist.sample(seed=key_sample_x_next)
            x_next = x_next.reshape((self.x_dim,))
        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(loc=x_next, scale=jnp.zeros_like(x_next)), new_dynamics_params

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        return DynamicsParams(key=key)


class LearnedCarSystem(System[DynamicsParams, CarRewardParams]):
    def __init__(self,
                 model: BatchedNeuralNetworkModel,
                 include_noise: bool,
                 predict_difference: bool,
                 **car_reward_kwargs: dict):
        reward = CarReward(**car_reward_kwargs)
        dynamics = LearnedDynamics(x_dim=reward.x_dim, u_dim=reward.u_dim, model=model, include_noise=include_noise,
                                   predict_difference=predict_difference)
        System.__init__(self, dynamics=dynamics, reward=CarReward(**car_reward_kwargs))

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
        new_key, key_x_next, key_reward = jr.split(system_params.key, 3)
        x_next_dist, next_dynamics_params = self.dynamics.next_state(x, u, system_params.dynamics_params)
        x_next = x_next_dist.sample(seed=key_x_next)
        reward_dist, next_reward_params = self.reward(x, u, system_params.reward_params, x_next)
        return SystemState(x_next=x_next,
                           reward=reward_dist.sample(seed=key_reward),
                           system_params=SystemParams(dynamics_params=next_dynamics_params,
                                                      reward_params=next_reward_params,
                                                      key=new_key),
                           )
