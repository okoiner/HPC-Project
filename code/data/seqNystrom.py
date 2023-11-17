import numpy as np
from scipy.linalg import norm, cholesky, qr, svd, solve_triangular
from data_generation import *

def randomized_nystrom(A, omega, k):
	C = A @ omega
	B = omega.T @ C
	try:
		L = cholesky(B, lower=True)
		Z = solve_triangular(L, C.T, lower=True).T
	except:
		B_U, B_S, B_Vt = svd(B, full_matrices=False)
		pseudo_SqrtS = np.array([1./B_S[i]**0.5 if B_S[i] != 0 else 0 for i in range(B_S.size)])
		Z = C @ B_U @ np.diag(pseudo_SqrtS) @ B_U.T
	
	Q, R = qr(Z, mode='economic')

	U, S, Vt = svd(R, full_matrices=False)
	U_k = U[:,:k]
	S_k = S[:k]

	Uhat_k = Q @ U_k

	return Uhat_k @ np.diag(S_k**2) @ Uhat_k.T

if __name__ == "__main__":
	n = 2**9
	l = 600
	k = 400

	A = A_YearPredictionMSD(n, 10**5)
	omega = np.random.randn(n, l)

	A_k = randomized_nystrom(A, omega, k)
	
	print(np.linalg.norm(A - A_k, ord='nuc')/np.linalg.norm(A, ord='nuc'))
