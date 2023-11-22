import numpy as np

def save_result(row):
	with open('data/test_results', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(row)

#date&time
#computer_owner
#n
#matrix_type
#R
#p
#sigma
#parallel/sequencial
#n processors
#l
#k
#sketch_matrix
#random_seed
#error_nuc
#runtime
