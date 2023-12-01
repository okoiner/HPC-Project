import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "../testing/" + "D_matrix1_error_vs_nz_vs_lk.csv"
table = pd.read_csv(file_path)

colors = ["red", "blue", "green", "magenta", "purple", "gray", "red", "blue", "green", "magenta", "purple", "gray", "red", "blue", "green", "magenta", "purple", "gray","red", "blue", "green", "magenta", "purple", "gray"]

kk = np.sort(list(set(table['k'])))
for i, k in enumerate(kk):
	rows = table[(table['k'] == k)].sort_values(by='k')
	errors = rows["error_nuc"]
	nzs = rows["nz"]
	plt.semilogy(nzs, errors, "-x", c = colors[i], label= "$k = %d$, " % k)
plt.xlabel("number of non-zeros per row")
plt.ylabel("Trace relative error")
plt.legend()
#plt.title("YearPredictionMSD, SRHT sketching\n$n = 4096$, $\sigma = 10^{5}$")
#plt.savefig("../plots/matrix3_error_vs_k_vs_l.pdf", format='pdf')
plt.show()
