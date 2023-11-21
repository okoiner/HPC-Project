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
	
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
s = comm.Get_size()

n = 2**10
l = 50
k = 30

A = None
if rank == 0:
	A = A_YearPredictionMSD(n, 1e6)
	wt = MPI.Wtime()

#========block distribution of A========
even_power = (int(math.log2(s))%2 == 0)

n_row_blocks, n_col_blocks = 2**math.ceil(math.log2(s)/2), 2**math.floor(math.log2(s)/2)
proc_row, proc_col = rank//n_col_blocks, rank%n_row_blocks

comm_row = comm.Split(color = proc_row, key = proc_col)
comm_col = comm.Split(color = proc_col, key = proc_row)

r_local, c_local = n//n_row_blocks, n//n_col_blocks
s_local = n//s

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
C_prod = A_local @ omega_local

if even_power:
	C_reduced = np.empty((r_local,l))
	comm_row.Reduce(C_prod, C_reduced, op = MPI.SUM, root = proc_row)
	
	C_local = np.empty((s_local,l))
	comm_col.Scatterv(C_reduced, C_local, root = proc_col)
	
	new_rank = proc_col*n_row_blocks + proc_row
	new_comm = comm.Split(color = 0, key = new_rank)
else:
	C_reduced = np.empty((r_local,l))
	comm_row.Reduce(C_prod, C_reduced, op = MPI.SUM, root = proc_row//2)
	
	temp_comm = comm_col.Split(color = proc_row%2, key = proc_row//2)
	
	C_local = np.empty((s_local,l))
	temp_comm.Scatterv(C_reduced, C_local)
	
	new_rank = proc_col*n_row_blocks + proc_row//2 + 4*(proc_row%2)
	new_comm = comm.Split(color = 0, key = new_rank)

B_local = omega_local[((new_rank%n_row_blocks)*c_local//n_row_blocks):((new_rank%n_row_blocks + 1)*c_local//n_row_blocks),:].T @ C_local
B = np.empty((l,l))
new_comm.Reduce(B_local, B, op = MPI.SUM, root=0)

#========cholesky========
cholesky_success = True
if new_rank == 0:
	try:
		L = cholesky(B, lower=True)
	except np.linalg.LinAlgError :
		print(":(")
		cholesky_success = False
		U_B, S_B, Vt_B = svd(B, full_matrices=False)
		S_pseudo_sqrt = np.array([1./s_b**0.5 if s_b != 0 else 0 for s_b in S_B])
		L_pseudo_inv = U_B @ np.diag(S_pseudo_sqrt) @ U_B.T

cholesky_success = new_comm.bcast(cholesky_success, root=0)
if cholesky_success:
	if rank != 0: L = np.empty((l,l))
	new_comm.Bcast(L, root=0)
	Z_local = solve_triangular(L, C_local.T, lower=True).T
else:
	if rank != 0: L_pseudo_inv = np.empty((l,l))
	new_comm.Bcast(L_pseudo_inv, root=0)
	Z_local = C_local @ L_pseudo_inv

#========TSQR========
toFactor = Z_local
Q_list = []

is_active = True
TSQR_steps = int(math.log2(s))
for step in range(TSQR_steps):
	if is_active:
		activeComm = new_comm.Split(color = 1 + new_rank//2**(step+1), key = new_rank)
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
		activeComm = new_comm.Split(color = 0, key = new_rank)
if new_rank == 0:
	(Q_step, R) = np.linalg.qr(toFactor, mode='reduced')
	Q_list.append(Q_step)

#========truncated SVD========
if new_rank == 0:
	U, S, Vt = svd(R, full_matrices=False)
	U_k = U[:,:k]
	S_k = S[:k]
else:
	S_k = np.empty(k, dtype = 'd')
#========Q product========
if new_rank == 0:
	Uhat_k_local = Q_step @ U_k

for step in reversed(range(TSQR_steps)):
	is_active = new_rank % (2**step) == 0
	if is_active:
		activeComm = comm.Split(color = 1 + new_rank/2**(step+1), key = new_rank)
		active_rank = activeComm.Get_rank()
		if active_rank == 0:
			activeComm.send(Uhat_k_local[l:,:], dest = 1)
			Uhat_k_local = Q_list[step] @ Uhat_k_local[0:l,:]
		else:
			Uhat_k_local = activeComm.recv(source = 0)
			Uhat_k_local = Q_list[step] @ Uhat_k_local
		
	else:
		activeComm = comm.Split(color = 0, key = new_rank)

#========final multiplication========
Uhat_k = np.empty((n,k))
comm.Gatherv(Uhat_k_local, Uhat_k, root = 0)

if new_rank == 0:
	A_nystrom = Uhat_k @ np.diag(S_k**2) @ Uhat_k.T
	
	print("finished")
	print(np.linalg.norm(A - A_nystrom, ord='nuc')/np.linalg.norm(A, ord='nuc'))

#new_comm.Bcast(S_k, root=0)
#A_nystrom_local = (Uhat_k_local * S_k) @ Uhat_k_local.T
#A_nystrom = np.empty((n,n))
#new_comm.Reduce(A_nystrom_local, A_nystrom, op = MPI.SUM, root = 0)

