import numpy as np
class Memory:
    def __init__(self, capacity, key_dimension, value_dimension):
        self.capacity = capacity
        self.states = np.zeros((capacity, key_dimension))
        self.values = np.zeros((capacity, value_dimension))

        self.curr_capacity = 0
        self.curr_ = 0
        self.lru = np.zeros(capacity)
        self.tm = 0

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

    def add(self, keys, values):

        for i, _ in enumerate(keys):
            self.curr_ = (self.curr_ + 1) % self.capacity
            self.states[self.curr_] = keys[i]
            self.values[self.curr_] = values[i]

            if self.curr_capacity < self.capacity:
                self.curr_capacity += 1

    def ran_add(self, keys, values):
        for i, key in enumerate(keys):
            if self.curr_capacity < self.capacity:
                self.curr_ = (self.curr_ + 1) % self.capacity
                self.states[self.curr_] = key
                self.values[self.curr_] = values[i]
            else:
                self.curr_ = np.random.choice(np.arange(self.curr_capacity), 1, replace=False)
                self.states[self.curr_] = key
                self.values[self.curr_] = values[i]
        # print("curr_capacity: {}".format(self.curr_capacity))

    def lru_add(self, keys, values):
        for i, key in enumerate(keys):
            if self.curr_capacity < self.capacity:
                self.curr_ = (self.curr_ + 1) % self.capacity
                self.states[self.curr_] = key
                self.values[self.curr_] = values[i]
            else:
                self.curr_ = np.argmin(self.lru)
                self.states[self.curr_] = key
                self.values[self.curr_] = values[i]