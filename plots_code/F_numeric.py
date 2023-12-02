import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "../testing_backup/" + "F_svd.csv"

titles = ["PolyDecay, $R = 10$, $p = 1$\n$\ell = 2k$ block SRHT", "ExpDecay, $R = 10$, $p = 0.25$\n$\ell = 2k$ block SRHT", "MNIST, $\sigma = 100$\n$\ell = 2k$ block SRHT", "YearPredictionMSD, $\sigma = 10^5$\n$\ell = 2k$ block SRHT"]
plot_names = ["F_PolyDecay", "F_ExpDecay", "F_MNIST", "F_Year_a"]
color_dict = ["red", "blue", "green", "magenta", "purple", "orange", "cyan", "pink", "gray", "brown", "black", "teal", "maroon", "olive", "navy", "lime", "coral", "beige", "turquoise", "indigo", "violet"]

table = pd.read_csv(file_path)
matrix_types = np.sort(list(set(table['matrix_type'])))
for type_id, matrix_type in enumerate(matrix_types):
	rows = table[table['matrix_type'] == matrix_type].sort_values(by='k')
	for index, row in rows.iterrows():
		sv = np.load("../data/singular_values/" + row['save_sv'])
		plt.plot(list(range(sv.shape[1])), sv[0]/sv[1], "-", c = color_dict[index], label="$k = %d$" % row['k'])
	plt.xlim((0, max(rows['k'])+3))
	plt.ylim((0, 1.05))
	plt.xlabel("singular value index")
	plt.ylabel("Error ratio $\sigma_i(A)/\sigma_i(A_k)$")
	plt.legend()
	plt.title(titles[type_id])
	plt.savefig("../plots/" + plot_names[type_id] + ".pdf", format='pdf')
	plt.close()
