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
	
	kk = [50, 100, 200, 400] #np.sort(list(set(table['k'])))
	sketchs = np.sort(list(set(table['sketch_matrix'])))
	for k_id, k in enumerate(kk):
		for sketch_id, sketch in enumerate(sketchs):
			rows = table[(table['k'] == k) & (table['sketch_matrix'] == sketch)].sort_values(by='k')
			errors = rows["error_nuc"]
			ratios_lk = rows["ratio_lk"]
			plt.semilogy(ratios_lk, errors, line_style_dict[sketch_id], c = color_dict[k_id], label="$k = %d$, " % k + "\t" + sketch_dict[sketch])
	plt.xlabel("ratio of $\ell$ and $k$")
	plt.ylabel("Trace relative error")
	plt.legend()
	plt.title(titles[file_id])
	plt.savefig("../plots/" + plot_names[file_id] + ".pdf", format='pdf')
	plt.close()
