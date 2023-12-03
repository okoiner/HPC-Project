import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

path_prefix = "../testing_backup/"
file_names = ["D_matrix1_error_vs_nz_vs_lk.csv",
	"D_matrix2_error_vs_nz_vs_lk.csv",
	"D_matrix3a_error_vs_nz_vs_lk.csv"]
titles = ["ExpDecay, $R = 10$, $p = 0.25$, $\ell = 2k$, short-axis", "MNIST, $\sigma = 100$, $\ell = 2k$, short-axis", "YearPredictionMSD, $\sigma = 10^5$, $\ell = 2k$, short-axis"]
plot_names = ["D_ExpDecay", "D_MNIST", "D_Year_a"]
sketch_dict = ["SRHT", "short-axis", "gaussian", "block SRHT"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]
line_style_dict = ["-x", "--+", ":*"]

legend_done = False
for file_id, file_name in enumerate(file_names):
	table = pd.read_csv(path_prefix + file_name)
	
	kk = np.sort(list(set(table['k'])))
	fig = plt.figure()
	for k_id, k in enumerate(kk):
		rows = table[table['k'] == k].sort_values(by='nz')
		errors = rows.groupby('nz')['error_nuc'].mean()
		nz = np.sort(list(set(rows['nz'])))
		plt.semilogy(nz, errors, line_style_dict[0], c = color_dict[k_id], label="$k = %d$, " % k)
	plt.xlabel("Number of non-zeros per row in $\Omega$", fontsize=14)
	plt.ylabel("Trace relative error", fontsize=14)
	plt.xticks(np.sort(list(set(table['nz']))), fontsize=12)
	plt.yticks(fontsize=12)
	plt.title(titles[file_id], fontsize=15)
	with PdfPages("../plots/" + plot_names[file_id] + ".pdf") as pdf:
		pdf.savefig(fig, bbox_inches='tight')
	
	if not legend_done:
		#save legend
		handles, labels = plt.gca().get_legend_handles_labels()
		fig_legend = plt.figure()
		fig_legend.legend(handles, labels, loc='center', ncol = 1)
		
		with PdfPages("../plots/D_legend.pdf") as pdf:
			pdf.savefig(fig_legend, bbox_inches='tight')
		legend_done = True
	
	plt.close('all')
