from os import environ
environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
import numpy as np
import math
import time
from scipy.linalg import norm, cholesky, qr, svd, solve_triangular
from data_generation import *

def strong_transpose(M):
	return np.array([np.copy(M[:,i]) for i in range(M.shape[1])])

def stopp(comm, rank):
	if rank == 0: input("continue?")
	dummy = 0
	dummy = comm.bcast(dummy, root=0)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
s = comm.Get_size()

n = 2**10
l = 200
k = 150

assert n > 0 and math.log2(n).is_integer() and int(math.log2(n))%2 == 0, "n must be a power of 4"
assert n//s >= l, "l is too big, TSQR will fail, change it to " + str(n//s) + " or less"
assert l >= k, "l must be greater or equal than k"

A = None
if rank == 0:
	A = A_YearPredictionMSD(n, 10**5)
	wt = MPI.Wtime()

#========block distribution of A========
n_rowcol = int(math.sqrt(s))
row, col = rank%n_rowcol, rank//n_rowcol	#counting along the columns turns out to be more "natural"

comm_row = comm.Split(color = row, key = col)
comm_col = comm.Split(color = col, key = row)

rc_local = n//n_rowcol
s_local = n//s

A_big_row_transp = None
if col == 0:
	A_big_row = np.empty((rc_local, n), dtype = 'd')
	comm_col.Scatterv(A, A_big_row, root=0)
	A_big_row_transp = strong_transpose(A_big_row)
A_local_transp = np.empty((rc_local, rc_local))
comm_row.Scatterv(A_big_row_transp, A_local_transp, root=0)
A_local = strong_transpose(A_local_transp)

#========block generation of omega (SRHT)========
general_random_seed = None
if rank == 0:
	general_random_seed = 42 #np.random.randint(2**30)
general_random_seed = comm.bcast(general_random_seed, root = 0)
col_random_seed = general_random_seed + col + 1
local_random_seed = general_random_seed + n_rowcol + rank + 1

general_rng = np.random.default_rng(general_random_seed)
col_rng = np.random.default_rng(col_random_seed)
local_rng = np.random.default_rng(local_random_seed)

randCol = general_rng.choice(n, l, replace=False)
signs = col_rng.choice([-1, 1], size=rc_local)
omega_local = np.fromfunction(np.vectorize(lambda i, j: signs[i]*(-1)**(bin((i + col*rc_local) & randCol[j]).count("1"))), (rc_local, l), dtype=int) / math.sqrt(l)

#========block multiplications========
C_prod = A_local @ omega_local
C_reduced = np.empty((rc_local,l))
comm_row.Reduce(C_prod, C_reduced, op = MPI.SUM, root = row)

C_local = np.empty((s_local,l))
comm_col.Scatterv(C_reduced, C_local, root = col)

B_local = omega_local[(row*s_local):((row+1)*s_local),:].T @ C_local
B = np.empty((l,l))
comm.Reduce(B_local, B, op = MPI.SUM, root=0)

#========cholesky========
cholesky_success = True
if rank == 0:
	try:
		L = strong_transpose(cholesky(B))
	except np.linalg.LinAlgError:
		print(":(")
		cholesky_success = False
		U_B, S_B, Vt_B = svd(B, full_matrices=False)
		S_pseudo_sqrt = np.array([1./s_b**0.5 if s_b < 1e-4 else 0 for s_b in S_B])
		L = (U_B * S_pseudo_sqrt) @ U_B.T
else:
	L = np.empty((l,l))
comm.Bcast(L, root=0)

cholesky_success = comm.bcast(cholesky_success, root=0)
if cholesky_success:
	Z_local = solve_triangular(L, C_local.T, lower=True).T
else:
	Z_local = C_local @ L

#========TSQR========
toFactor = Z_local
Q_list = []

toFactorrr = np.empty((n,l))
comm.Gatherv(toFactor, toFactorrr, root = 0)
if rank == 0: np.save("parallel", toFactorrr)

is_active = True
TSQR_steps = int(math.log2(s))
for step in range(TSQR_steps):
	if is_active:
		activeComm = comm.Split(color = 1 + rank//2**(step+1), key = rank)
		active_rank = activeComm.Get_rank()
		(Q_step, R_step) = np.linalg.qr(toFactor, mode='reduced')
		Q_list.append(Q_step)
		
		if active_rank == 0:
			R_rec = activeComm.recv(source = 1)
			toFactor = np.vstack((R_step, R_rec))
		else:
			activeComm.send(R_step, dest = 0)
			is_active = False
	else:
		activeComm = comm.Split(color = 0, key = rank)
if rank == 0:
	(Q_step, R) = np.linalg.qr(toFactor, mode='reduced')
	Q_list.append(Q_step)

#========truncated SVD========
if rank == 0:
	U, S, Vt = svd(R, full_matrices=False)
	U_k = U[:,:k]
	S_k = S[:k]
else:
	S_k = np.empty(k, dtype = 'd')
#========Q product========
if rank == 0:
	Uhat_k_local = Q_step @ U_k

for step in reversed(range(TSQR_steps)):
	is_active = rank % (2**step) == 0
	if is_active:
		activeComm = comm.Split(color = 1 + rank/2**(step+1), key = rank)
		active_rank = activeComm.Get_rank()
		if active_rank == 0:
			activeComm.send(Uhat_k_local[l:,:], dest = 1)
			Uhat_k_local = Q_list[step] @ Uhat_k_local[0:l,:]
		else:
			Uhat_k_local = activeComm.recv(source = 0)
			Uhat_k_local = Q_list[step] @ Uhat_k_local
		
	else:
		activeComm = comm.Split(color = 0, key = rank)

#========final multiplication========
Uhat_k = np.empty((n,k))
comm.Gatherv(Uhat_k_local, Uhat_k, root = 0)

if rank == 0:
	A_nystrom = Uhat_k @ np.diag(S_k**2) @ Uhat_k.T
	
	print("finished")
	print(np.linalg.norm(A - A_nystrom, ord='nuc')/np.linalg.norm(A, ord='nuc'))

#new_comm.Bcast(S_k, root=0)
#A_nystrom_local = (Uhat_k_local * S_k) @ Uhat_k_local.T
#A_nystrom = np.empty((n,n))
#new_comm.Reduce(A_nystrom_local, A_nystrom, op = MPI.SUM, root = 0)

