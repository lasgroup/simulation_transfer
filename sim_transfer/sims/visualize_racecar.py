from sim_transfer.sims.simulators import RaceCarSim
import jax.numpy as jnp
import jax
from matplotlib import pyplot as plt

sim_lf = RaceCarSim(use_blend=False)
sim_hf = RaceCarSim(use_blend=True, car_id=2)


INIT_STATE = jnp.array(
    [[-9.5005625e-01, -1.4144412e+00, 9.9892426e-01, 4.6371352e-02, 7.2260178e-04, 8.1058703e-03, -7.7542849e-03],
     [-1, -0.5, 0., 1.0, 0.5, 0.5, 1.],
     [0.5, -1.5, 0., -2.0, -0.5, -0.5, 1.],])


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
            if k == 0:
                a = jnp.array([- 1 * jnp.sin(2 * t), 0.8 / (t + 1)])
            elif k == 1:
                a = jnp.array([+ 1 * jnp.sin(4 * t), 0.8 / (t + 1)])
            elif k == 2:
                a = jnp.array([- 1, 0.8 / (t + 1)])
            a = jnp.repeat(a[None, :], NUM_PARALLEL, axis=0)
            x = jnp.concatenate([s, a], axis=-1)
            s = fun_stacked(x)
            traj.append(s)
            actions.append(a)

        traj = jnp.stack(traj, axis=0)
        actions = jnp.stack(actions, axis=0)
        from matplotlib import pyplot as plt

        for i in range(NUM_PARALLEL):
            axes[k][j].plot(traj[:, i, 0], traj[:, i, 1])
        axes[k][j].set_xlim(-1, 1.)
        axes[k][j].set_ylim(-2, 1.)

    axes[k][2].plot(jnp.arange(len(actions[:, 0, 0])), actions[:, 0, 0])
fig.show()