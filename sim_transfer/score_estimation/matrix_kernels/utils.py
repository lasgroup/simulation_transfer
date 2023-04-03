import jax
import jax.numpy as jnp

def median_heuristic(x, y):
    # x: [..., n, d]
    # y: [..., m, d]
    # return: []
    n = jnp.shape(x)[-2]
    m = jnp.shape(y)[-2]
    x_expand = jnp.expand_dims(x, -2)
    y_expand = jnp.expand_dims(y, -3)
    pairwise_dist = jnp.sqrt(jnp.sum(jnp.square(x_expand - y_expand), axis=-1))
    k = n * m // 2
    top_k_values = jax.lax.top_k(
        jnp.reshape(pairwise_dist, [-1, n * m]),
        k=k)[0]
    kernel_width = jnp.reshape(top_k_values[:, -1], jnp.shape(x)[:-2])
    return jax.lax.stop_gradient(kernel_width)
