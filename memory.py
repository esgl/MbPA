import numpy as np
class Memory:
    def __init__(self, capacity, key_dimension, value_dimension):
        self.capacity = capacity
        self.states = np.zeros((capacity, key_dimension))
        self.values = np.zeros((capacity, value_dimension))

        self.curr_capacity = 0
        self.curr_ = 0

    def sample(self, n_samples):
        if self.curr_capacity < n_samples:
            idx = np.random.choice(np.arange(len(self.states)), n_samples, replace=False)
        else:
            idx = np.random.choice(np.arange(self.curr_capacity), n_samples, replace=False)
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

        # print("curr_capacity: {}".format(self.curr_capacity))
