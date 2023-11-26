import numpy as np
import csv
import datetime

def set_counter(n, name = "counter"):
	np.save("data/counter_" + name + ".npy", np.array(int(n), dtype=int))

def get_counter(name = "counter"):
	return np.load("data/counter_" + name + ".npy")

def add_counter(n, name = "counter"):
	set_counter(get_counter(name) + n, name)

def settings_from_csv(line_id):
	with open("data/test_results.csv", newline='') as csvfile:
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
	use_SRHT = False if row[11] == '0' else True
	kk = int(row[12]) if len(row[12]) > 0 else 0
	
	return n, matrix_type, R, p, sigma, l, k, use_SRHT, kk

def print_settings(n, matrix_type, R, p, sigma, l, k, use_SRHT, kk, n_processors):
	out = "Settings: n = " + str(n) + "\nmatrix_type: "
	match matrix_type:
		case 0:
			out = out + "PolyDecay, R = " + str(R) + ", p = " + str(p) + "\n"
		case 1:
			out = out + "ExpDecay, R = " + str(R) + ", p = " + str(p) + "\n"
		case 2:
			out = out + "A_MNIST, sigma = " + str(sigma) + "\n"
		case 3:
			out = out + "YearPredictionMSD, sigma = " + str(sigma) + "\n"
	out = out + "sketch " + ("SRHT" if use_SRHT else "short-axis") + ", l = " + str(l) + (", kk = " + str(kk) if not use_SRHT else "") + "\nk = " + str(k) + ", in " + ("parallel with " + str(n_processors) + " processors" if n_processors > 1 else " sequential")
	print(out)

def read_computer_name():
    with open("data/computer_name.txt", 'r', encoding='utf-8') as file:
        computer_name = file.read()
    return computer_name[:-1]

def save_results_to_csv(line_id, n_processors, random_seed, error_nuc, wt):
	with open("data/test_results.csv", newline='') as file:
		reader = csv.reader(file)
		data = list(reader)

	data[line_id][0] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	data[line_id][1] = read_computer_name()
	data[line_id][2] = str(n_processors)
	data[line_id][3] = "par" if n_processors > 1 else "seq"
	data[line_id][13]= str(random_seed)
	data[line_id][14]= str(error_nuc)
	data[line_id][15]= str(wt)

	with open("data/test_results.csv", 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(data)

def print_results(error_nuc, wt):
	print("error_nuc = " + str(error_nuc) + ", runtime = " + str(wt))

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

#n
#matrix_type
#R
#p
#sigma
#l
#k
#sketch_type
