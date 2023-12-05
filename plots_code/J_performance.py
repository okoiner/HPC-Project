import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

special_k = 200

path_prefix = "../testing_discarded/"
file_names = ["B_matrix2_error_vs_k_vs_l.csv",
	"B_seq_matrix2_error_vs_k_vs_l.csv"]
plot_names = ["J_SRHT", "J_short_axis", "J_gaussian", "J_block_SRHT"]
titles_dic = ["SRHT", "sparse-short-axis sketching, k = " + str(special_k), "gaussian", "block SRHT sketching, k = " + str(special_k)]
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
	for ps_id, ps in enumerate(parseqs):
		rows = table[(table['sketch_matrix'] == sketch) & (table['k'] == special_k) & (table['parallel_sequencial'] == ps)].sort_values(by='l')
		runtimes = rows["runtime"]
		ll = rows["l"]
		plt.semilogy(ll, runtimes, line_style_dict[0], c = color_dict[ps_id], label=("parallel" if ps == "par" else "sequential"))
	plt.xlabel("Sketch size (k)", fontsize=14)
	plt.ylabel("Runtime", fontsize=14)
	plt.xticks(ll, fontsize=12)
	plt.yticks(fontsize=12)
	plt.title(titles_dic[sketch], fontsize=15)
	with PdfPages("../plots/" + plot_names[sketch] + ".pdf") as pdf:
		pdf.savefig(fig, bbox_inches='tight')
	
	if not legend_done:
		#save legend
		handles, labels = plt.gca().get_legend_handles_labels()
		fig_legend = plt.figure()
		fig_legend.legend(handles, labels, loc='center', ncol = 1)
		
		with PdfPages("../plots/J_legend.pdf") as pdf:
			pdf.savefig(fig_legend, bbox_inches='tight')
		legend_done = True

	plt.close('all')
