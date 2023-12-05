import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

path_prefix = "../testing_discarded/"
file_names = ["B_matrix2_error_vs_k_vs_l.csv",
	"B_seq_matrix2_error_vs_k_vs_l.csv"]
plot_names = ["I_SRHT", "I_short_axis", "I_gaussian", "I_block_SRHT"]
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
	ratioss = []
	for l_id, l in enumerate(ll):
		rows = table[(table['sketch_matrix'] == sketch) & (table['l'] == l)].sort_values(by='k')
		ratios = rows[rows['parallel_sequencial'] == 'seq']["runtime"].to_numpy(dtype='d')/rows[rows['parallel_sequencial'] == 'par']["runtime"].to_numpy(dtype='d')
		ratioss = ratioss + [ratios[0]]
	plt.semilogy(ll, ratioss, line_style_dict[0], c = color_dict[0], label="$\ell = %d$, " % l)
	
	plt.xlabel("sketch size (l)", fontsize=14)
	plt.ylabel("sequential runtime/parallel runtime", fontsize=14)
	plt.xticks(ll, fontsize=12)
	plt.yticks(fontsize=12)
	plt.title(titles_dic[sketch], fontsize=15)
	with PdfPages("../plots/" + plot_names[sketch] + ".pdf") as pdf:
		pdf.savefig(fig, bbox_inches='tight')
	
	plt.close('all')
