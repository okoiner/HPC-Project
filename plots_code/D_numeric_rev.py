import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_prefix = "../testing_backup/"
file_names = ["D_matrix1_error_vs_nz_vs_lk.csv",
	"D_matrix2_error_vs_nz_vs_lk.csv",
	"D_matrix3a_error_vs_nz_vs_lk.csv"]
titles = ["ExpDecay, $R = 10$, $p = 0.25$,\n$\ell = 2k$, short-axis", "MNIST, $\sigma = 100$,\n$\ell = 2k$, short-axis", "YearPredictionMSD, $\sigma = 10^5$,\n$\ell = 2k$, short-axis"]
plot_names = ["D_ExpDecay", "D_MNIST", "D_Year_a"]
sketch_dict = ["SRHT", "short-axis", "gaussian", "block SRHT"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]
line_style_dict = ["-x", "--+", ":*"]

for file_id, file_name in enumerate(file_names):
	table = pd.read_csv(path_prefix + file_name)
	
	nzs = np.sort(list(set(table['nz'])))
	for nz_id, nz in enumerate(nzs):
		rows = table[table['nz'] == ].sort_values(by='nz')
		errors = rows.groupby('nz')['error_nuc'].mean()
		nz = np.sort(list(set(rows['nz'])))
		plt.semilogy(nz, errors, line_style_dict[0], c = color_dict[k_id], label="$k = %d$, " % k)
	plt.xlabel("number of non-zeros per row in $\Omega$")
	plt.ylabel("Trace relative error")
	plt.legend()
	plt.title(titles[file_id])
	plt.savefig("../plots/" + plot_names[file_id] + ".pdf", format='pdf')
	plt.close()
