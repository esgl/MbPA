import numpy as np
from annoy import AnnoyIndex

class LRU_KNN:
    def __init__(self, capacity, key_dim, value_dim, batch_size):
        self.capacity = capacity
        self.curr_capacity = 0

        self.states = np.zeros((capacity, key_dim))
        self.values = np.zeros((capacity, value_dim))
        self.lru = np.zeros(capacity)
        self.tm = 0.0

        self.index = AnnoyIndex(key_dim, metric="euclidean")
        self.index.set_seed(123)

        self.initial_update_size = batch_size
        self.min_update_size = self.initial_update_size
        self.cached_states = []
        self.cached_values = []
        self.cached_indices = []

    def nn(self, keys, k):
        dists = []
        inds = []
        for key in keys:
            ind, dist = self.index.get_nns_by_vector(key, k, include_distances=True)
            dists.append(dist)
            inds.append(ind)
        return dists, inds

    def query(self, keys, k):
        _, indices = self.nn(keys, k)
        states = []
        values = []

        for ind in indices:
            self.lru[ind] = self.tm
            states.append(self.states[ind])
            values.append(self.values[ind])
        self.tm += 0.001
        return states, values

    def _insert(self, keys, values, indices):
        self.cached_states = self.cached_states + keys
        self.cached_values = self.cached_values + values
        self.cached_indices = self.cached_indices + indices

        if len(self.cached_states) >= self.min_update_size:
            self.min_update_size = max(self.initial_update_size, self.curr_capacity * 0.02)
            self._update_index()

    def _update_index(self):
        self.index.unbuild()
        for i, ind in enumerate(self.cached_indices):
            new_state = self.cached_states[i]
            new_value = self.cached_values[i]

            self.states[ind] = new_state
            self.values[ind] = new_value
            self.index.add_item(ind, new_state)

        self.cached_states = []
        self.cached_values = []
        self.cached_indices = []

        self.index.build(50)
        self.built_capacity = self.curr_capacity

    def _rebuild_index(self):
        self.index.unbuild()
        for ind, state in enumerate(self.states[:self.curr_capacity]):
            self.index.add_item(ind, state)
        self.index.build(50)
        self.built_capacity = self.curr_capacity

