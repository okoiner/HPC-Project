import numpy as np
import data_generation as dg
from sketching import *
import time

#dg.A_MNIST(2**13)

l = 100

for p in range(7, 16):
	wt1 = time.time()
	M1 = SRHT_sketch(2**p, l)
	wt1 = time.time() - wt1
	wt2 = time.time()
	M2 = SRHT_sketch_fast(2**p, l)
	wt2 = time.time() - wt2
	
	print(wt1, wt2, np.linalg.norm(M1-M2))
