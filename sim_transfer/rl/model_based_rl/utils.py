import chex
import jax.numpy as jnp
from jax import random


def split_data(x: chex.Array, y: chex.Array, key: chex.PRNGKey, test_ratio=0.2):
    """
    Splits the data into training and test sets.
    Parameters:
        x (array): Input data.
        y (array): Output data.
        key (PRNGKey): Random key.
        test_ratio (float): Fraction of the data to be used as test data.
    Returns:
        x_train, x_test, y_train, y_test
    """
    n = x.shape[0]
    idx = jnp.arange(n)
    permuted_idx = random.permutation(key, idx)
    test_size = int(n * test_ratio)
    train_idx = permuted_idx[:-test_size]
    test_idx = permuted_idx[-test_size:]

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = jnp.array([1, 0, 1, 0, 1])

    x_train, x_test, y_train, y_test = split_data(x, y, test_ratio=0.4, seed=42)

    print("x_train:", x_train)
    print("x_test:", x_test)
    print("y_train:", y_train)
    print("y_test:", y_test)
