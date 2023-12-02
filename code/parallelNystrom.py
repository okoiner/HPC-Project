from os import environ
environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
import numpy as np
import math
import time
from scipy.linalg import norm, cholesky, qr, svd, solve_triangular
from data_generation import *
from utility import *

def strong_transpose(M):
	return np.array([np.copy(M[:,i]) for i in range(M.shape[1])])

def SRHT(n, l, col, rc_local, general_random_seed, col_random_seed):
	general_rng = np.random.default_rng(general_random_seed)
	col_rng = np.random.default_rng(col_random_seed)

	randCol = general_rng.choice(n, l, replace=False)
	signs = col_rng.choice([-1, 1], size=rc_local)
	return np.fromfunction(np.vectorize(lambda i, j: signs[i]*(-1)**(bin((i + col*rc_local) & randCol[j]).count("1"))), (rc_local, l), dtype=int) / math.sqrt(l)

def block_short_axis(l, col, rc_local, col_random_seed, nz):
	col_rng = np.random.default_rng(col_random_seed)
	
	sketch = np.zeros((rc_local, l), dtype='d')
	bounds = np.ceil(np.linspace(0,l,nz+1))
	for i in range(rc_local):
		 col = col_rng.integers(bounds[:nz], bounds[1:], size=nz)
		 sketch[i,col] = col_rng.choice([-1, 1], size=nz)*col_rng.uniform(1., 2., size=nz)
	return sketch

def block_gaussian(l, rc_local, col_random_seed):
	col_rng = np.random.default_rng(col_random_seed)
	return col_rng.normal(size = (rc_local, l))

def block_SRHT(l, rc_local, general_random_seed, col_random_seed):
	general_rng = np.random.default_rng(general_random_seed)
	col_rng = np.random.default_rng(col_random_seed)
	
	randCol = general_rng.choice(n, l, replace=False)
	signsRows = col_rng.choice([-1, 1], size=rc_local)
	signsCols = col_rng.choice([-1, 1], size=l)
	
	return np.fromfunction(np.vectorize(lambda i, j: signsRows[i]*signsCols[j]*(-1)**(bin(i & randCol[j]).count("1"))), (rc_local, l), dtype=int) / math.sqrt(l)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
s = comm.Get_size()

#In the following section we read from a csv file the settings for the test
save_results = True
save_svd = True
(n,l,k,sketch_matrix,nz) = (None, None, None, None, None)
if rank == 0:
	print("")
	line_id = get_counter()
	n, matrix_type, RR, p, sigma, l, k, sketch_matrix, nz = get_settings_from_csv(line_id)
	print_settings(n, matrix_type, RR, p, sigma, l, k, sketch_matrix, nz, s)
(n,l,k,sketch_matrix,nz) = comm.bcast((n,l,k,sketch_matrix,nz), root = 0)

#assert n > 0 and math.log2(n).is_integer() and int(math.log2(n))%2 == 0, "n must be a power of 4"
assert n//s >= l, "l is too big, TSQR will fail, change it to " + str(n//s) + " or less"
assert l >= k, "l must be greater or equal than k"
assert nz <= l, "nz must be smaller or equal than l"

A = None
if rank == 0:
	match matrix_type:
		case 0:
			A = A_PolyDecay(n, RR, p)
		case 1:
			A = A_ExpDecay(n, RR, p)
		case 2:
			A = A_MNIST(n, sigma)
		case 3:
			A = A_YearPredictionMSD(n, sigma)
		case _:
			raise Exception("Unknown matrix type")

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
	general_random_seed = np.random.randint(2**30)
general_random_seed = comm.bcast(general_random_seed, root = 0)
col_random_seed = general_random_seed + col + 1
#local_random_seed = general_random_seed + n_rowcol + rank + 1

match sketch_matrix:
	case 0:
		omega_local = SRHT(n, l, col, rc_local, general_random_seed, col_random_seed)
	case 1:
		omega_local = block_short_axis(l, col, rc_local, col_random_seed, nz)
	case 2:
		omega_local = block_gaussian(l, rc_local, col_random_seed)
	case 3:
		omega_local = block_SRHT(l, rc_local, general_random_seed, col_random_seed)
	case _:
		raise Exception("Unknown sketch type")

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
		S_pseudo_sqrt = np.array([(1./s_b)**0.5 if s_b != 0 else 0. for s_b in S_B], dtype = 'd')
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

#========final gather========
Uhat_k = np.empty((n,k))
comm.Gatherv(Uhat_k_local, Uhat_k, root = 0)

if rank == 0:
	wt = MPI.Wtime() - wt
	print("lowrank approximation completed")
	
	#we keep the following multiplication outside the runtime because normally it doesn't make sense to compute it
	A_nystrom = Uhat_k @ np.diag(S_k**2) @ Uhat_k.T
	
	if save_svd:
		_, realS, _ = svd(A, full_matrices=False)
		save_svd_to_csv(line_id, S_k**2, realS[:k])
	
	error_nuc = np.linalg.norm(A - A_nystrom, ord='nuc')/nuc_norm_A(matrix_type, n, RR, p, sigma)
	if save_results:
		save_results_to_csv(line_id, s, cholesky_success, general_random_seed, error_nuc, wt)
		add_counter(1)
	print_results(error_nuc, wt, cholesky_success, general_random_seed)
