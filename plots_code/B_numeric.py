import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_prefix = "../testing/"
file_names = ["B_matrix0_error_vs_k_vs_l.csv",
	"B_matrix1_test.csv",
	"B_matrix2_error_vs_k_vs_l.csv",
	"B_matrix3a_error_vs_k_vs_l.csv",
	"B_matrix3b_error_vs_k_vs_l.csv"]
titles = ["PolyDecay, $R = 10$, $p = 1$", "ExpDecay, $R = 10$, $p = 0.25$", "MNIST, $\sigma = 100$", "YearPredictionMSD, $\sigma = 10^5$", "YearPredictionMSD, $\sigma = 10^6$"]
plot_names = ["B_PolyDecay", "B_ExpDecay", "B_MNIST", "B_Year_a", "B_Year_b"]
sketch_dict = ["SRHT", "short-axis", "gaussian", "block SRHT"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]
line_style_dict = ["-x", "--+", ":*"]

for file_id, file_name in enumerate(file_names):
	table = pd.read_csv(path_prefix + file_name)
	
	ll = np.sort(list(set(table['l'])))
	sketchs = np.sort(list(set(table['sketch_matrix'])))
	for l_id, l in enumerate(ll):
		for sketch_id, sketch in enumerate(sketchs):
			rows = table[(table['l'] == l) & (table['sketch_matrix'] == sketch)].sort_values(by='k')
			errors = rows["error_nuc"]
			kk = rows["k"]
			plt.semilogy(kk, errors, line_style_dict[sketch_id], c = color_dict[l_id], label="$\ell = %d$, " % l + "\t" + sketch_dict[sketch])
	plt.xlabel("Approximation (k)")
	plt.ylabel("Trace relative error")
	plt.legend()
	plt.title(titles[file_id])
	plt.savefig("../plots/" + plot_names[file_id] + ".pdf", format='pdf')
	plt.close()
