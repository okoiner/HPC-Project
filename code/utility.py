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

"""
def save_results_to_csv(line_id, n_processors, random_seed, error_nuc, wt)
	with open("data/test_results.csv", newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    data[line_id][0] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data[line_id][1] = "stefano"
    data[line_id][2] = str(n_processors)
    data[line_id][3] = "par" if n_processors > 1 else "seq"
    data[line_id][13]= str(random_seed)
	data[line_id][14]= str(error_nuc)
	data[line_id][15]= str(wt)

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
"""
	
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

#n
#matrix_type
#R
#p
#sigma
#l
#k
#sketch_type
