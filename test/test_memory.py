from memory import Memory
import numpy as np
key_dimension = 3
capacity = 10
value_dimension = 2

m = Memory(capacity, key_dimension, value_dimension)

keys = np.random.randn(capacity, key_dimension)
values = np.random.randint(0, 1, (capacity, value_dimension))

print(keys)
print(values)
print(values[0][0])
print(type(values[0][0]))
m.add(keys,values)

print(m.sample(capacity))
key, value = m.sample(capacity)
print(value[0][0])
print(type(value[0][0]))