from sim_transfer.sims.simulators import RaceCarSim
import jax.numpy as jnp
import jax
from matplotlib import pyplot as plt
from sim_transfer.sims.envs import RCCarSimEnv
from sim_transfer.sims.util import decode_angles, encode_angles

sim_lf = RaceCarSim(use_blend=False, car_id=2)
sim_hf = RaceCarSim(use_blend=True, car_id=2)


INIT_STATE = jnp.array(
    [[-9.5005625e-01, -1.4144412e+00, 9.9892426e-01, 4.6371352e-02, 7.2260178e-04, 8.1058703e-03, -7.7542849e-03],
     [-1, -0.5, 0., 1.0, 0.5, 0.5, 1.],
     [0.5, -1.5, 0., -2.0, -0.5, -0.5, 1.],])

ACTIONS = [lambda t: jnp.array([- 1 * jnp.sin(2 * t), 0.8 / (t + 1)]),
           lambda t: jnp.array([+ 1 * jnp.sin(4 * t), 0.8 / (t + 1)]),
           lambda t: jnp.array([- 1, 0.8 / (t + 1)])]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
for k in range(3):
    for j, sim in enumerate([sim_lf, sim_hf]):
        key = jax.random.PRNGKey(435345)
        NUM_PARALLEL = 10
        fun_stacked = sim.sample_functions(num_samples=NUM_PARALLEL, rng_key=key)
        fun_stacked = jax.jit(fun_stacked)

        s = jnp.repeat(INIT_STATE[k][None, :], NUM_PARALLEL, axis=0)
        traj = [s]
        actions = []
        for i in range(30):
            t = i / 30.
            a = ACTIONS[k](t)
            a = jnp.repeat(a[None, :], NUM_PARALLEL, axis=0)
            x = jnp.concatenate([s, a], axis=-1)
            s = fun_stacked(x)
            traj.append(s)
            actions.append(a)

        traj = jnp.stack(traj, axis=0)
        actions = jnp.stack(actions, axis=0)

        for i in range(NUM_PARALLEL):
            axes[k][j].plot(traj[:, i, 0], traj[:, i, 1])
        axes[k][j].set_xlim(-1, 1.)
        axes[k][j].set_ylim(-2, 1.)

        env = RCCarSimEnv(encode_angle=True, use_obs_noise=False, use_tire_model=bool(j))
        obs = env.reset()
        env._state = decode_angles(INIT_STATE[k], angle_idx=2)
        traj_env = [encode_angles(env._state, angle_idx=2)]
        for i in range(30):
            t = i / 30.
            a = ACTIONS[k](t)
            obs, _, _, _ = env.step(a)
            traj_env.append(obs)
        traj_env = jnp.stack(traj_env, axis=0)
        axes[k][j].plot(traj_env[:, 0], traj_env[:, 1], color='black', linewidth=3)

    axes[k][2].plot(jnp.arange(len(actions[:, 0, 0])), actions[:, 0, 0])
fig.show()