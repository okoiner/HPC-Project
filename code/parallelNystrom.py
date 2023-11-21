from os import environ
environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
import numpy as np
import math
from scipy.linalg import norm, cholesky, qr, svd, solve_triangular
from data_generation import *

def strong_transpose(M):
	return np.array([np.copy(M[:,i]) for i in range(M.shape[1])])
	
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
s = comm.Get_size()

n = 2**2
l = 2
k = 200

A = None
if rank == 0:
	A = np.arange(n**2, dtype = 'd').reshape((n,n))
	print(A)
	wt = MPI.Wtime()

#========block distribution of A========
n_row_blocks, n_col_blocks = 2**math.ceil(math.log2(n)/2), 2**math.floor(math.log2(n)/2)
proc_row, proc_col = rank//n_col_blocks, rank%n_row_blocks

comm_row = comm.Split(color = proc_row, key = proc_col)
comm_col = comm.Split(color = proc_col, key = proc_row)

r_local, c_local = n//n_row_blocks, n//n_col_blocks

A_big_row_transp = None
if proc_col == 0:
	A_big_row = np.empty((r_local, n), dtype = 'd')
	comm_col.Scatterv(A, A_big_row, root=0)
	A_big_row_transp = strong_transpose(A_big_row)
A_local_transp = np.empty((r_local, c_local))
comm_row.Scatterv(A_big_row_transp, A_local_transp, root=0)
A_local = strong_transpose(A_local_transp)

#========block generation of omega (SRHT)========
general_random_seed = None
if rank == 0:
	general_random_seed = np.random.randint(2**30)
	print(general_random_seed)
general_random_seed = comm.bcast(general_random_seed, root = 0)
col_random_seed = general_random_seed + proc_col + 1
local_random_seed = general_random_seed + n_col_blocks + rank + 1

general_rng = np.random.default_rng(general_random_seed)
col_rng = np.random.default_rng(col_random_seed)
local_rng = np.random.default_rng(local_random_seed)

randCol = general_rng.choice(n, l, replace=False)
signs = col_rng.choice([-1, 1], size=n)
omega_local = np.fromfunction(np.vectorize(lambda i, j: signs[i]*(-1)**(bin(i & randCol[j]).count("1"))), (c_local, l), dtype=int) / math.sqrt(l)

#========block multiplications========
C_local = A_local @ omega_local
C_row = np.empty((r_local,l))
comm_row.Reduce(C_local, C_row, op = MPI.SUM, root = 0)
C = np.empty((n, l))
comm.Gatherv(C_row, C, root = 0)
B_local = omega_local.T @ C_local
B = np.empty((l,l))
comm.Reduce(B_local, B, op = MPI.SUM, root = 0)

#========cholesky========
if rank == 0:
	try:
		L = cholesky(B, lower=True)
		Z = solve_triangular(L, C.T, lower=True).T
	except:
		B_U, B_S, B_Vt = svd(B, full_matrices=False)
		pseudo_sqrtS = np.array([1./b_s**0.5 if b_s != 0 else 0 for b_s in B_S])
		Z = C @ B_U @ np.diag(pseudo_sqrtS) @ B_U.T

#========TSQR========
s_local = n//s
toFactor = np.empty((s_local, l))
comm_row.Scatterv(C_row, toFactor)

Q_list = []

is_active = True
n_steps = int(math.log2(s))
for step in range(n_steps):
	if is_active:
		activeComm = comm.Split(color = 1 + rank/2**(step+1), key = rank)
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
	Q_local = Q_step


