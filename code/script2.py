import numpy as np
import data_generation as dg
from sketching import *
import time

try:
	np.linalg.cholesky(np.array([[0,0],[0,-1/0]]))
except np.linalg.LinAlgError:
	print("ciao")
except:
	print(":(")
