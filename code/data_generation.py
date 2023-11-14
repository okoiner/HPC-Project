import numpy as np

def A_PolyDecay(n, R, p):
	return np.diag(np.array([1]*R + [x**(-p) for x in range(2,n-R+2)]))
