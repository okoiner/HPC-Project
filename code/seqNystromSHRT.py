import numpy as np
from scipy.linalg import norm, cholesky, qr, svd, solve_triangular
from data_generation import *
from sketching import *

def randomized_nystrom(A, omega, k):
	C = A @ omega
	B = omega.T @ C
	
	try:
		L = cholesky(B, lower=True)
		Z = solve_triangular(L, C.T, lower=True).T
	except np.linalg.LinAlgError:
		print(":(")
		B_U, B_S, B_Vt = svd(B, full_matrices=False)
		pseudo_sqrtS = np.array([1./b_s**0.5 if b_s != 0 else 0 for b_s in B_S])
		Z = C @ B_U @ np.diag(pseudo_sqrtS) @ B_U.T
	
	Q, R = qr(Z, mode='economic')

	U, S, Vt = svd(R, full_matrices=False)
	U_k = U[:,:k]
	S_k = S[:k]

	Uhat_k = Q @ U_k

	return Uhat_k @ np.diag(S_k**2) @ Uhat_k.T

if __name__ == "__main__":
	n = 2**10
	l = 64
	k = 32

	A = A_YearPredictionMSD(n, 10**5)
	
	general_random_seed = 42 #np.random.randint(2**30)
	general_rng = np.random.default_rng(general_random_seed)
	randCol = general_rng.choice(n, l, replace=False)
	
	col_random_seed_0 = general_random_seed + 1
	col_random_seed_1 = general_random_seed + 1 + 1

	col_rng_0 = np.random.default_rng(col_random_seed_0)
	col_rng_1 = np.random.default_rng(col_random_seed_1)
	signs = np.concatenate((col_rng_0.choice([-1, 1], size=n//2), col_rng_1.choice([-1, 1], size=n//2)))
	
	omega = np.fromfunction(np.vectorize(lambda i, j: signs[i]*(-1)**(bin(i & randCol[j]).count("1"))), (n, l), dtype=int) / math.sqrt(l)
	
	A_k = randomized_nystrom(A, omega, k)
	
	print(np.linalg.norm(A - A_k, ord='nuc')/np.linalg.norm(A, ord='nuc'))
