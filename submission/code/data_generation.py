import numpy as np

def A_PolyDecay(n, R, p):
	return np.diag(np.array([1]*R + [x**(-p) for x in range(2,n-R+2)]))

def A_ExpDecay(n, R, p):
	return np.diag(np.array([1]*R + [10**(-x*p) for x in range(1,n-R+1)], dtype = 'd'))

def A_MNIST(n, sigma = 100):
	try:
		return np.load("../data/generated_matrices/A_MNIST_" + str(n) + "_" + str(sigma) + ".npy")
	except FileNotFoundError:
		print("The matrix has to be generated, this can take time")
		
		X = np.load("../data/dataMNIST.npy")/255.
		m = X.shape[1]
		
		sigma2 = sigma**2
		A = np.fromfunction(lambda i, j: np.exp(- sum(((X[i, d] - X[j, d])**2) for d in range(m))/sigma2), (n, n), dtype=int)
		
		np.save("../data/generated_matrices/A_MNIST_" + str(n) + "_" + str(sigma) + ".npy", A)
		
		print("Generation complete")
		
		return A

def A_YearPredictionMSD(n, sigma):
	try:
		return np.load("../data/generated_matrices/A_YearPredictionMSD_" + str(n) + "_" + str(sigma) + ".npy")
	except FileNotFoundError:
		print("The matrix has to be generated, this can take time")
		
		X = np.load("../data/dataYearPredictionMSDreduced.npy")
		m = X.shape[1]
		
		sigma2 = sigma**2
		A = np.fromfunction(lambda i, j: np.exp(- sum(((X[i, d] - X[j, d])**2) for d in range(m))/sigma2), (n, n), dtype=int)
		
		np.save("../data/generated_matrices/A_YearPredictionMSD_" + str(n) + "_" + str(sigma) + ".npy", A)
		
		print("Generation complete")
		
		return A
