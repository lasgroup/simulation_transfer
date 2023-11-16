from typing import List, Iterable, Tuple

import jax
import jax.numpy as jnp

from .util import RngKeyMixin


class DataLoader(RngKeyMixin):

    def __init__(self, x_data: jnp.ndarray, y_data: jnp.ndarray, rng_key: jax.random.PRNGKey,
                 batch_size: int = 1, shuffle: bool = False, drop_last: bool = False):
        super().__init__(rng_key)

        assert x_data.shape[0] == y_data.shape[0]
        self.x_data = x_data
        self.y_data = y_data
        self.num_data_points = x_data.shape[0]

        # check batch size
        assert batch_size > 0 or batch_size == -1
        self.batch_size = batch_size if batch_size > 0 else self.num_data_points
        assert batch_size <= self.num_data_points

        self.shuffle = shuffle
        self.drop_last = drop_last

    def _split_indices(self, indices: List[int]):
        assert len(indices) == self.num_data_points
        num_splits_full = len(indices) // self.batch_size
        idx_splits = jnp.split(indices[:(num_splits_full * self.batch_size)], num_splits_full)
        num_remaining_points = len(indices) % self.batch_size
        if not self.drop_last and num_remaining_points > 0:
            idx_splits.append(indices[-num_remaining_points:])
        assert sum(map(lambda l: len(l), idx_splits)) == len(indices) - \
               int(self.drop_last) * len(indices) % self.batch_size
        return idx_splits

    def _get_data_by_indices(self, indices) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.x_data[indices], self.y_data[indices]

    def __iter__(self) -> Iterable:
        _indices = jnp.arange(self.num_data_points)
        if self.shuffle:
            _indices = jax.random.permutation(self.rng_key, _indices)
        _indices_batches = self._split_indices(_indices)

        def data_iter():
            for batch_idx in _indices_batches:
                yield self._get_data_by_indices(batch_idx)

        return iter(data_iter())
