import numpy as np

def A_PolyDecay(n, R, p):
	return np.diag(np.array([1]*R + [x**(-p) for x in range(2,n-R+2)]))

def A_ExpDecay(n, R, p):
	return np.diag(np.array([1]*R + [10**(-x*p) for x in range(1,n-R+1)]))

def A_MNIST(n, sigma = 100):
	try:
		return np.load("A_MNIST_" + str(n) + "_" + str(sigma) + ".npy")
	except:
		X = np.load("dataMNIST.npy")
		m = X.shape[1]
		
		sigma2 = sigma**2
		A = np.fromfunction(lambda i, j: np.exp(- sum(((X[i, d] - X[j, d])**2) for d in range(m))/sigma2), (n, n), dtype=int)
		
		np.save("A_MNIST_" + str(n) + "_" + str(sigma) + ".npy", A)
		return A

def A_YearPredictionMSD(n, sigma):
	try:
		return np.load("A_YearPredictionMSD_" + str(n) + "_" + str(sigma) + ".npy")
	except:
		X = np.load("dataYearPredictionMSDreduced.npy")
		m = X.shape[1]
		
		sigma2 = sigma**2
		A = np.fromfunction(lambda i, j: np.exp(- sum(((X[i, d] - X[j, d])**2) for d in range(m))/sigma2), (n, n), dtype=int)
		
		np.save("A_YearPredictionMSD_" + str(n) + "_" + str(sigma) + ".npy", A)
		return A
