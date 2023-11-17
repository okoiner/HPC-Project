import numpy as np
from seqNystrom import *
from data_generation import *
import matplotlib.pyplot as plt

n = 2**11
A = A_YearPredictionMSD(n, 10**5)
Anuc = np.linalg.norm(A, ord='nuc')

ll = np.array([400 + 100*i for i in range(5)])
kk = np.array([100*i for i in range(1,11)])
err = np.empty((ll.size, kk.size))

for j in range(ll.size):
	omega = np.random.randn(n, ll[j])
	for i in range(kk.size):
		err[j,i] = np.linalg.norm(A - randomized_nystrom(A, omega, kk[i]), ord='nuc')/Anuc

for row in err:
	plt.semilogy(kk, row)
plt.show()
