import numpy as np
import csv
import datetime
from data_generation import *

def set_counter(n, name = "default"):
	np.save("../data/utilities/counter_" + name + ".npy", np.array(int(n), dtype=int))

def get_counter(name = "default"):
	return np.load("../data/utilities/counter_" + name + ".npy")

def add_counter(n, name = "default"):
	set_counter(get_counter(name) + n, name)

def get_computer_name():
    with open("../data/utilities/computer_name.txt", 'r', encoding='utf-8') as file:
        computer_name = file.read()
    return computer_name[:-1]
    
def get_test_name():
    with open("../data/utilities/test_name.txt", 'r', encoding='utf-8') as file:
        test_name = file.read()
    return test_name

def get_settings_from_csv(line_id, file_name = None):
	if file_name == None:
		file_name = get_test_name()
	print("Getting settings from line " + str(line_id) + " in " + file_name)
	with open("../testing/" + file_name, newline='') as csvfile:
		reader = csv.reader(csvfile)
		for i, row in enumerate(reader):
			if i == line_id:
				break;
	if len(row[0]) > 0:
		raise Exception("test already executed")
	
	n = 2**int(row[4])
	matrix_type = int(row[5])
	R = int(row[6]) if len(row[6]) > 0 else 0
	p = float(row[7]) if len(row[7]) > 0 else 0
	sigma = int(row[8]) if len(row[8]) > 0 else 0
	l = int(row[9])
	k = int(row[10])
	sketch_matrix = int(row[11])
	t = int(row[12]) if len(row[12]) > 0 else 0
	
	return n, matrix_type, R, p, sigma, l, k, sketch_matrix, t

def print_settings(n, matrix_type, R, p, sigma, l, k, sketch_matrix, t, n_processors):
	output = "Settings: n = " + str(n) + "\nMatrix_type: "
	match matrix_type:
		case 0:
			output = output + "PolyDecay, R = " + str(R) + ", p = " + str(p) + "\n"
		case 1:
			output = output + "ExpDecay, R = " + str(R) + ", p = " + str(p) + "\n"
		case 2:
			output = output + "A_MNIST, sigma = " + str(sigma) + "\n"
		case 3:
			output = output + "YearPredictionMSD, sigma = " + str(sigma) + "\n"
	sketch_dict = {0: "SRHT", 1: "short-axis", 2: "gaussian", 3:"block_SRHT"}
	output = output + "sketch " + (sketch_dict[sketch_matrix]) + ", l = " + str(l) + (", t = " + str(t) if sketch_matrix == 1 else "") + "\nk = " + str(k) + ", in " + ("parallel with " + str(n_processors) + " processors" if n_processors > 1 else "sequential")
	print(output)

def save_results_to_csv(line_id, n_processors, cholesky_success, random_seed, error_nuc, wt, file_name = None):
	if file_name == None:
		file_name = get_test_name()
	
	with open("../testing/" + file_name, newline='') as file:
		reader = csv.reader(file)
		data = list(reader)

	data[line_id][0] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	data[line_id][1] = get_computer_name()
	data[line_id][2] = str(n_processors)
	data[line_id][3] = "par" if n_processors > 1 else "seq"
	data[line_id][13]= 1 if cholesky_success else 0
	data[line_id][14]= str(random_seed)
	data[line_id][15]= str(error_nuc)
	data[line_id][16]= str(wt)

	with open("../testing/" + file_name, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(data)

def save_svd_to_csv(line_id, approx_sv, real_sv, file_name = None):
	print("Saving singular values to csv")
	
	unique_id = get_counter("singular_values")
	np.save("../data/singular_values/sv_" + str(unique_id).zfill(4) + ".npy", np.array([approx_sv, real_sv]))
	
	if file_name == None:
		file_name = get_test_name()
	
	with open("../testing/" + file_name, newline='') as file:
		reader = csv.reader(file)
		data = list(reader)
	
	data[line_id][17]= "sv_" + str(unique_id).zfill(3) + ".npy"

	with open("../testing/" + file_name, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(data)
	
	add_counter(1, "singular_values")

def nuc_norm_A(matrix_type, n, R, p, sigma):
	match matrix_type:
		case 0:
			try:
				return np.load("../data/nuclear_norms/nuc_norm_PolyDecay_" + str(n) + "_" + str(R) + "_" + str(p) + ".npy")
			except FileNotFoundError:
				nuc_norm = np.linalg.norm(A_PolyDecay(n, R, p), ord='nuc')
				np.save("../data/nuclear_norms/nuc_norm_PolyDecay_" + str(n) + "_" + str(R) + "_" + str(p) + ".npy", nuc_norm)
				return nuc_norm
		case 1:
			try:
				return np.load("../data/nuclear_norms/nuc_norm_ExpDecay_" + str(n) + "_" + str(R) + "_" + str(p) + ".npy")
			except FileNotFoundError:
				nuc_norm = np.linalg.norm(A_ExpDecay(n, R, p), ord='nuc')
				np.save("../data/nuclear_norms/nuc_norm_ExpDecay_" + str(n) + "_" + str(R) + "_" + str(p) + ".npy", nuc_norm)
				return nuc_norm
		case 2:
			try:
				return np.load("../data/nuclear_norms/nuc_norm_MNIST_" + str(n) + "_" + str(sigma) + ".npy")
			except FileNotFoundError:
				nuc_norm = np.linalg.norm(A_MNIST(n, sigma), ord='nuc')
				np.save("../data/nuclear_norms/nuc_norm_MNIST_" + str(n) + "_" + str(sigma) + ".npy", nuc_norm)
				return nuc_norm
		case 3:
			try:
				return np.load("../data/nuclear_norms/nuc_norm_YearPredictionMSD_" + str(n) + "_" + str(sigma) + ".npy")
			except FileNotFoundError:
				nuc_norm = np.linalg.norm(A_YearPredictionMSD(n, sigma), ord='nuc')
				np.save("../data/nuclear_norms/nuc_norm_YearPredictionMSD_" + str(n) + "_" + str(sigma) + ".npy", nuc_norm)
				return nuc_norm

def print_results(error_nuc, wt, cholesky_success, random_seed):
	print("error_nuc = " + str(error_nuc) + ", runtime = " + str(wt))
	print("cholesky " + ("succeded" if cholesky_success else "failed") + ", random_seed = " + str(random_seed))
