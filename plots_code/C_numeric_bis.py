import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_prefix = "../testing_backup/"
file_names = ["C_matrix0_error_vs_k_vs_lratio.csv",
	"C_matrix2_error_vs_k_vs_lratio.csv",
	"C_matrix3a_error_vs_k_vs_lratio.csv"]
titles = ["ExpDecay, $R = 10$, $p = 0.25$", "MNIST, $\sigma = 100$", "YearPredictionMSD, $\sigma = 10^5$"]
plot_names = ["C_ExpDecay", "C_MNIST", "C_Year_a"]
sketch_dict = ["SRHT", "short-axis", "gaussian", "block SRHT"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]
line_style_dict = ["-x", "--+", ":*"]

for file_id, file_name in enumerate(file_names):
	table = pd.read_csv(path_prefix + file_name)
	
	ratios_lk = [1, 1.25, 1.5, 2, 3] #np.sort(list(set(table['ratio_lk'])))
	sketchs = np.sort(list(set(table['sketch_matrix'])))
	for ratio_id, ratio_lk in enumerate(ratios_lk):
		for sketch_id, sketch in enumerate(sketchs):
			rows = table[(table['ratio_lk'] == ratio_lk) & (table['sketch_matrix'] == sketch)].sort_values(by='k')
			errors = rows["error_nuc"]
			kk = rows["k"]
			plt.semilogy(kk, errors, line_style_dict[sketch_id], c = color_dict[ratio_id], label="$\ell = %dk$, " % ratio_lk + "\t" + sketch_dict[sketch])
	plt.xlabel("k")
	plt.ylabel("Trace relative error")
	plt.legend(ncol = 2, fontsize='small')
	plt.title(titles[file_id])
	plt.savefig("../plots/" + plot_names[file_id] + "_bis.pdf", format='pdf')
	plt.close()
