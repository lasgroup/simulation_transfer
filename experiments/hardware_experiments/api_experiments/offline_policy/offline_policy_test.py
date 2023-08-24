import pickle
import jax.numpy as jnp
import numpy as np
from mbpo.optimizers.policy_optimizers.sac.sac_networks import SACNetworksModel, make_inference_fn
from brax.training.acme import running_statistics
import flax.linen as nn
import jax.random as jr
from sim_transfer.hardware.car_env import CarEnv

ENCODE_ANGLE = True

normalize_fn = running_statistics.normalize
sac_networks_model = SACNetworksModel(
    x_dim=7, u_dim=2,
    preprocess_observations_fn=normalize_fn,
    policy_hidden_layer_sizes=(64, 64),
    policy_activation=nn.swish,
    critic_hidden_layer_sizes=(64, 64),
    critic_activation=nn.swish)

inference_fn = make_inference_fn(sac_networks_model.get_sac_networks())

with open('params.pkl', 'rb') as file:
    params = pickle.load(file)


def policy(obs):
    dummy_obs = obs[0:7]
    return np.asarray(inference_fn(params, deterministic=True)(dummy_obs, jr.PRNGKey(0))[0])

env = CarEnv(encode_angle=ENCODE_ANGLE)
obs, _ = env.reset()
for i in range(200):
    action = policy(obs)
    obs, reward, terminate, info = env.step(action)
    print(obs)
env.close()
