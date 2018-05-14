import numpy as np
from annoy import AnnoyIndex

class Memory:
    def __init__(self, capacity, state_dim, value_dim):
        self.capacity = capacity
        print("state_dim:", state_dim)
        self.states = np.zeros((capacity, state_dim))
        self.values = np.zeros((capacity, value_dim))

        self.curr_capacity = 0
        self.curr_ = 0
        self.lru = np.zeros(capacity)
        self.tm = 0

        self.cached_states = []
        self.cached_values = []
        self.cached_indices = []

        self.index = AnnoyIndex(state_dim)
        self.index.set_seed(123)
        self.update_size = 1000
        self.build_capacity = 0

    def sample_knn_test(self, state, k):
        inds, dists = self.index.get_nns_by_vector(state, k, include_distances=True)
        return self.states[inds], self.values[inds], dists





    def sample_knn(self, states, k):
        dists = []
        inds = []
        for state in states:
            ind, dist = self.index.get_nns_by_vector(state, k, include_distances=True)
            inds.append(ind)
            dists.append(dist)
        # inds = np.reshape(np.array(inds), -1)
        return self.states[inds], self.values[inds], dists

    def sample(self, n_samples):
        if self.curr_capacity < n_samples or n_samples == 0:
            idx = np.random.choice(np.arange(len(self.states)), n_samples, replace=False)
        else:
            idx = np.random.choice(np.arange(self.curr_capacity), n_samples, replace=False)
        self.tm += 0.01
        self.lru[idx] = self.tm
        embs = self.states[idx]
        values = self.values[idx]

        return embs, values


    def add_knn(self, states, values):
        self._add_knn(states, values)

    def add_knn_lru(self, states, values):
        self._add_knn(states, values, lru=True)

    def add(self, states, values):
        self._add(states, values)

    def add_lru(self, states, values):
        self._add(states, values, lru=True)

    def add_rand(self, states, values):
        self._add(states, values, rand=True)

    def _insert(self, states, values, indices):
        self.cached_states = self.cached_states + states
        self.cached_values = self.cached_values + values
        self.cached_indices = self.cached_indices + indices
        if len(self.cached_states) >= self.update_size:
            self._update_index()

    def _update_index(self):
        self.index.unbuild()
        for i, ind in enumerate(self.cached_indices):
            self.states[ind] = self.cached_states[i]
            self.values[ind] = self.cached_values[i]
            self.index.add_item(ind, self.cached_states[i])

        self.index.build(50)
        self.build_capacity = self.curr_capacity

        self.cached_states = []
        self.cached_values = []
        self.cached_indices = []

    def _rebuild_index(self):
        self.index.unbuild()
        for ind, state in enumerate(self.states[:self.curr_capacity]):
            self.index.add_item(ind, state)
        self.index.build(50)
        self.build_capacity = self.curr_capacity

    def _add_knn(self, states, values, lru=False):
        # print(states)
        indices = []
        states_ = []
        values_ = []
        for i, _ in enumerate(states):
            if lru:
                if self.curr_capacity >= self.capacity:
                    ind = np.argmin(self.lru)
                else:

                    ind = self.curr_capacity
                    self.curr_capacity += 1
            else:
                if self.curr_capacity >= self.capacity:
                    self.curr_ = (self.curr_ + 1) % self.capacity
                    ind = self.curr_
                else:
                    ind = self.curr_capacity
                    self.curr_capacity += 1

            self.lru[ind] = self.tm
            indices.append(ind)
            states_.append(states[i])
            values_.append(values[i])
        self._insert(states_, values_, indices)

    def _add(self, states, values, rand=False, lru=False):
        for i, state in enumerate(states):
            if self.curr_capacity < self.capacity:
                self.curr_ = (self.curr_ + 1) % self.capacity
                self.states[self.curr_] = state
                self.values[self.curr_] = values[i]
                if self.curr_capacity < self.capacity:
                    self.curr_capacity += 1
            else:
                if lru:
                    self.curr_ = np.argmin(self.lru)
                if rand:
                    self.curr_ = np.random.choice(np.arange(self.curr_capacity), 1, replace=False)

                if not lru and not rand:
                    self.curr_ = (self.curr_ + 1) % self.capacity
                self.states[self.curr_] = state
                self.values[self.curr_] = values[i]
    @property
    def length(self):
        # assert self.index.get_n_items() == self.curr_capacity
        # return self.curr_capacity
        return  self.index.get_n_items()