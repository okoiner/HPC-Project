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
	omega = short_axis_sketch(n, l, 8)

	A_k = randomized_nystrom(A, omega, k)
	
	print(np.linalg.norm(A - A_k, ord='nuc')/np.linalg.norm(A, ord='nuc'))
