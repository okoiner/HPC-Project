import counter as ct
import numpy as np

l = 1002
k = 13

bounds = np.ceil(np.linspace(0,l,k+1))

print(np.random.randint(bounds[:k], bounds[1:], size=k))
