import numpy as np

def set_counter(n, name = "counter"):
	np.save("data/counter_" + name + ".npy", np.array(int(n), dtype=int))

def get_counter(name = "counter"):
	return np.load("data/counter_" + name + ".npy")

def add_counter(n, name = "counter"):
	set_counter(get_counter(name) + n, name)
