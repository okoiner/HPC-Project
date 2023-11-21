import numpy as np
from seqNystrom import *
from data_generation import *
from sketching import *
import matplotlib.pyplot as plt

n = 2**10
A = A_MNIST(n, 100)
Anuc = np.linalg.norm(A, ord='nuc')

ll = np.array([400 + 100*i for i in range(5)])
kk = np.array([100*i for i in range(1,11)])
err = np.empty((ll.size, kk.size))

for j in range(ll.size):
	omega = SRHT_sketch(n, ll[j])
	for i in range(kk.size):
		err[j,i] = np.linalg.norm(A - randomized_nystrom(A, omega, kk[i]), ord='nuc')/Anuc

for row in err:
	plt.semilogy(kk, row)
plt.show()
