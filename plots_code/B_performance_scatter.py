import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_prefix = "../testing_backup/"
file_names = ["B_matrix0_error_vs_k_vs_l.csv",
	"B_matrix1_error_vs_k_vs_l.csv",
	"B_matrix2_error_vs_k_vs_l.csv",
	"B_matrix3a_error_vs_k_vs_l.csv",
	"B_matrix3b_error_vs_k_vs_l.csv"]
titles = ["PolyDecay, $R = 10$, $p = 1$", "ExpDecay, $R = 10$, $p = 0.25$", "MNIST, $\sigma = 100$", "YearPredictionMSD, $\sigma = 10^5$", "YearPredictionMSD, $\sigma = 10^6$"]
plot_names = ["B_dots_PolyDecay", "B_dots_ExpDecay", "B_dots_MNIST", "B_dots_Year_a", "B_dots_Year_b"]
sketch_dict = ["SRHT", "short-axis", "gaussian", "block SRHT"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]
line_style_dict = ["-x", "--+", ":*"]

for file_id, file_name in enumerate(file_names):
	table = pd.read_csv(path_prefix + file_name)
	
	sketchs = np.sort(list(set(table['sketch_matrix'])))
	for sketch_id, sketch in enumerate(sketchs):
		rows = table[table['sketch_matrix'] == sketch]
		errors = rows['error_nuc']
		runtimes = rows['runtime']
		plt.scatter(errors, runtimes, c = color_dict[sketch_id], label = sketch_dict[sketch])
	
	plt.xlabel("error")
	plt.ylabel("Runtime")
	plt.legend()
	plt.title(titles[file_id])
	plt.savefig("../plots/" + plot_names[file_id] + ".pdf", format='pdf')
	plt.close()
