import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

path_prefix = "../testing_discarded/"
file_names = ["B_matrix2_error_vs_k_vs_l.csv",
	"B_seq_matrix2_error_vs_k_vs_l.csv"]
plot_names = ["G_SRHT", "G_short_axis", "G_gaussian", "G_block_SRHT"]
titles_dic = ["SRHT", "sparse-short-axis sketching", "gaussian", "block SRHT sketching"]
sketch_dict = ["SRHT", "short-axis", "gaussian", "block SRHT"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]
line_style_dict = ["-x", "--+", ":*"]

tables = [pd.read_csv(path_prefix + file_name) for file_name in file_names]
table = pd.concat(tables, ignore_index=True)
sketchs = np.sort(list(set(table['sketch_matrix'])))

legend_done = False
for sketch_id, sketch in enumerate(sketchs):
	ll = np.sort(list(set(table['l'])))
	parseqs = np.sort(list(set(table['parallel_sequencial'])))
	fig = plt.figure()
	for l_id, l in enumerate(ll):
		for ps_id, ps in enumerate(parseqs):
			rows = table[(table['sketch_matrix'] == sketch) & (table['l'] == l) & (table['parallel_sequencial'] == ps)].sort_values(by='k')
			runtimes = rows["runtime"]
			kk = rows["k"]
			plt.semilogy(kk, runtimes, line_style_dict[ps_id], c = color_dict[l_id], label="$\ell = %d$, " % l + "\tin " + ("parallel" if ps == "par" else "sequential"))
	plt.xlabel("Approximation (k)", fontsize=14)
	plt.ylabel("Runtime", fontsize=14)
	plt.xticks([50, 200, 400, 600, 800, 1000], fontsize=12)
	plt.yticks([10**(-0.5), 1, 10**0.5], ["$10^{-0.5}$", "$10^0$", "$10^{0.5}$"], fontsize=12)
	plt.title(titles_dic[sketch], fontsize=15)
	with PdfPages("../plots/" + plot_names[sketch] + ".pdf") as pdf:
		pdf.savefig(fig, bbox_inches='tight')
	
	if not legend_done:
		#save legend
		handles, labels = plt.gca().get_legend_handles_labels()
		fig_legend = plt.figure()
		fig_legend.legend(handles, labels, loc='center', ncol = 1)
		
		with PdfPages("../plots/G_legend.pdf") as pdf:
			pdf.savefig(fig_legend, bbox_inches='tight')
		legend_done = True

	plt.close('all')
