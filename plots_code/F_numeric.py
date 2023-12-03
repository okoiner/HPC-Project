import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "../testing/" + "F_svd_v3.csv"

titles = ["PolyDecay, $R = 10$, $p = 1$,\n$\ell = 2k$, block SRHT", "ExpDecay, $R = 10$, $p = 0.25$,\n$\ell = 2k$, block SRHT", "MNIST, $\sigma = 100$,\n$\ell = 2k$, block SRHT", "YearPredictionMSD, $\sigma = 10^5$,\n$\ell = 2k$, block SRHT", "YearPredictionMSD, $\sigma = 10^6$,\n$\ell = 2k$, block SRHT"]
plot_names = ["F_PolyDecay", "F_ExpDecay", "F_MNIST", "F_Year_a", "F_Year_b"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]

count = 3

table = pd.read_csv(file_path)
matrix_types = np.sort(list(set(table['matrix_type'])))
for type_id, matrix_type in enumerate(matrix_types):
	rows = table[table['matrix_type'] == matrix_type].sort_values(by='k')
	kk = np.sort(list(set(rows['k'])))
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
	if matrix_type == 1 or matrix_type == 1:
		plt.ylim((0, 1.7))
	else:
		plt.ylim((0, 1.05))
	plt.xlabel("singular value index")
	plt.ylabel("Error ratio $\sigma_i(A)/\sigma_i(A_k)$")
	plt.legend()
	plt.title(titles[type_id])
	plt.savefig("../plots/" + plot_names[type_id] + ".pdf", format='pdf')
	plt.close()
