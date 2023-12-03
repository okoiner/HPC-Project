import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

path_prefix = "../testing_backup/"
file_names = ["C_matrix0_error_vs_k_vs_lratio.csv",
	"C_matrix2_error_vs_k_vs_lratio.csv",
	"C_matrix3a_error_vs_k_vs_lratio.csv"]
titles = ["ExpDecay, $R = 10$, $p = 0.25$", "MNIST, $\sigma = 100$", "YearPredictionMSD, $\sigma = 10^5$"]
plot_names = ["C_ExpDecay", "C_MNIST", "C_Year_a"]
sketch_dict = ["SRHT", "short-axis", "gaussian", "block SRHT"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]
line_style_dict = ["-x", "--+", ":*"]

legend_done = False
for file_id, file_name in enumerate(file_names):
	table = pd.read_csv(path_prefix + file_name)
	
	kk = [50, 100, 200, 400] #np.sort(list(set(table['k'])))
	sketchs = np.sort(list(set(table['sketch_matrix'])))
	fig = plt.figure()
	for k_id, k in enumerate(kk):
		for sketch_id, sketch in enumerate(sketchs):
			rows = table[(table['k'] == k) & (table['sketch_matrix'] == sketch)].sort_values(by='k')
			errors = rows["error_nuc"]
			ratios_lk = rows["ratio_lk"]
			plt.semilogy(ratios_lk, errors, line_style_dict[sketch_id], c = color_dict[k_id], label="$k = %d$, " % k + "\t" + sketch_dict[sketch])
	plt.xlabel("$\ell$/$k$ ratio", fontsize=12)
	plt.ylabel("Trace relative error", fontsize=12)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.title(titles[file_id], fontsize=15)
	with PdfPages("../plots/" + plot_names[file_id] + ".pdf") as pdf:
		pdf.savefig(fig, bbox_inches='tight')
	
	if not legend_done:
		#save legend
		handles, labels = plt.gca().get_legend_handles_labels()
		fig_legend = plt.figure()
		fig_legend.legend(handles, labels, loc='center', ncol = 1)
		
		with PdfPages("../plots/C_legend.pdf") as pdf:
			pdf.savefig(fig_legend, bbox_inches='tight')
		legend_done = True

	plt.close('all')
	
