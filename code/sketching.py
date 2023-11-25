import numpy as np
import math
from scipy.linalg import hadamard

def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()

def gaussian_sketch(n, l):
	return np.random.randn(n, l)

def walsh_hadamard_matrix_non_recursive(n):
    if n == 1:
        return np.array([[1]])

    H = np.empty((n, n))

    for i in range(n):
        for j in range(n):
            H[i, j] = (-1)**(bin(i & j).count("1"))

    return H

def SRHT_sketch_slow(n, l):
	if not is_power_of_two(n):
		raise ValueError("n must be a power of 2")
	
	D = np.diag(np.random.choice([-1, 1], size=n))
	H = hadamard(n)
	DH = D @ H
	randCol = np.random.choice(n, l, replace=False)
	DHRt = DH[:, randCol]

	return DHRt / math.sqrt(l)

def SRHT_sketch(n, l):
	if not is_power_of_two(n):
		raise ValueError("n must be a power of 2")
	
	signs = np.random.choice([-1, 1], size=n)
	randCol = np.random.choice(n, l, replace=False)
	
	return np.fromfunction(np.vectorize(lambda i, j: signs[i]*(-1)**(bin(i & randCol[j]).count("1"))), (n, l), dtype=int) / math.sqrt(l)

def short_axis_sketch_naive(n, l, k):
	sketch = np.zeros((n,l), dtype='d')
	for i in range(n):
		col = np.random.choice(l, k, replace=False)
		sketch[i,col] = np.random.choice([-1, 1], size=k)*np.random.uniform(1., 2., size=k)
	return sketch

def short_axis_sketch(n, l, k):
	sketch = np.zeros((n,l), dtype='d')
	bounds = np.ceil(np.linspace(0,l,k+1))
	for i in range(n):
		 col = np.random.randint(bounds[:k], bounds[1:], size=k)
		 sketch[i,col] = np.random.choice([-1, 1], size=k)*np.random.uniform(1., 2., size=k)
	return sketch
