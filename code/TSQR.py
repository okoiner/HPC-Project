from os import environ
environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
import math
from commonFunctions import *

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
s = comm.Get_size()

#==================== Matrix generation ====================

(m, n) = (None, None)
W = None

if rank == 0:
	print("generating W")
	
	m, n = 50000, 600
	matrixType = 1
	cond = 1e14
	save = False
	
	W = genMatrix(m, n, matrixType, cond)
	
	wt = MPI.Wtime()

#==================== Distribution ====================

(m,n) = comm.bcast((m, n), root = 0)
s_local = m//s
toFactor = np.empty((s_local, n))
comm.Scatterv(W, toFactor, root=0)

#==================== QR factorization ====================

if rank == 0: print("starting factorization")

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
			toFactor = np.vstack((R_step,R_rec))
		else:
			activeComm.send(R_step, dest = 0)
			is_active = False
	else:
		activeComm = comm.Split(color = 0, key = rank)

if rank == 0:
	(Q_step, R) = np.linalg.qr(toFactor, mode='reduced')
	Q_list.append(Q_step)

	Q_local = Q_step

for step in range(n_steps):
	is_active = rank % (2**(n_steps-step-1)) == 0
	if is_active:
		activeComm = comm.Split(color = 1 + rank/2**(n_steps-step), key = rank)
		active_rank = activeComm.Get_rank()
		if active_rank == 0:
			activeComm.send(Q_local[n:,:], dest = 1)
			Q_local = Q_list[n_steps-step-1] @ Q_local[0:n,:]
		else:
			Q_local = activeComm.recv(source = 0)
			Q_local = Q_list[n_steps-step-1] @ Q_local
		
	else:
		activeComm = comm.Split(color = 0, key = rank)

Q = np.empty((m,n))
comm.Gatherv(Q_local, Q, root = 0)

if rank == 0:
	wt = MPI.Wtime()-wt
	saveTestOnFile(save, "data.csv", s, "TSQR", wt, W, matrixType, Q, R)
