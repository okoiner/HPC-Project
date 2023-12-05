import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

file_path = "../testing/" + "F_svd.csv"

titles = ["PolyDecay, $R = 10$, $p = 1$, $\ell = 2k$, block SRHT", "ExpDecay, $R = 10$, $p = 0.25$, $\ell = 2k$, block SRHT", "MNIST, $\sigma = 100$, $\ell = 2k$, block SRHT", "YearPredictionMSD, $\sigma = 10^5$, $\ell = 2k$, block SRHT", "YearPredictionMSD, $\sigma = 10^6$, $\ell = 2k$, block SRHT"]
plot_names = ["F_PolyDecay", "F_ExpDecay", "F_MNIST", "F_Year_a", "F_Year_b"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]
ylims = [(0.25, 1.0375), (0, 1.7), (0.25, 1.0375), (0.8, 1.01), (0.8, 1.01)]

count = 3

table = pd.read_csv(file_path)
matrix_types = np.sort(list(set(table['matrix_type'])))
legend_done = False
for type_id, matrix_type in enumerate(matrix_types):
	rows = table[table['matrix_type'] == matrix_type].sort_values(by='k')
	kk = np.sort(list(set(rows['k'])))
	fig = plt.figure()
	for k_id, k in enumerate(reversed(kk)):
		rows2 = rows[rows['k'] == k]
		label_present = False
		
		counter = 0
		for index, row in rows2.iterrows():
			sv = np.load("../data/singular_values/" + row['save_sv'])
			if label_present:
				plt.plot(list(range(sv.shape[1])), sv[0]/sv[1], "-", c = color_dict[2-k_id], linewidth=1)
			else:
				plt.plot(list(range(sv.shape[1])), sv[0]/sv[1], "-", c = color_dict[2-k_id], label="$k = %d$" % row['k'], linewidth=1)
				label_present = True
			
			counter += 1
			if counter >= count:
				break
	plt.xlim((0, max(rows['k'])+3))
	plt.ylim(ylims[matrix_type])
	plt.xlabel("Singular value index", fontsize=14)
	plt.ylabel("Error ratio $\sigma_i(A_k)/\sigma_i(A)$", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.title(titles[type_id], fontsize=15)
	with PdfPages("../plots/" + plot_names[type_id] + ".pdf") as pdf:
		pdf.savefig(fig, bbox_inches='tight')
	
	if not legend_done:
		#save legend
		handles, labels = plt.gca().get_legend_handles_labels()
		fig_legend = plt.figure()
		fig_legend.legend(handles, labels, loc='center', ncol = 1)
		
		with PdfPages("../plots/F_legend.pdf") as pdf:
			pdf.savefig(fig_legend, bbox_inches='tight')
		legend_done = True

	plt.close('all')
