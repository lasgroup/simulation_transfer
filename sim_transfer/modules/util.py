import jax
import jax.numpy as jnp
from typing import List, Dict

def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list, treedef_list  = list(zip(*[jax.tree_flatten(tree) for tree in trees]))
    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jax.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


class RngKeyMixin:

    def __init__(self, rng_key: jax.random.PRNGKey):
        self._rng_key = rng_key

    def _next_rng_key(self) -> jax.random.PRNGKey:
        new_key, self._rng_key = jax.random.split(self._rng_key)
        return new_key

    @property
    def rng_key(self) -> jax.random.PRNGKey:
        return self._next_rng_key()


""" Stats aggregation """

def aggregate_stats(stats_list: List[Dict]) -> Dict:
    return jax.tree_map(jnp.mean, tree_stack(stats_list))