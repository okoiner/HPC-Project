import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "../testing/" + "matrix3_error_vs_k_vs_l.csv"
table = pd.read_csv(file_path)

ll = np.sort(list(set(table['l'])))
for i, l in enumerate(ll):
	rows = table[table['l'] == l].sort_values(by='k')
	errors = rows["error_nuc"]
	kk = rows["k"]
	plt.semilogy(kk, errors, "-x",label="$\ell = %d$" % l)
plt.xlabel("Approximation (k)")
plt.ylabel("Trace relative error")
plt.legend()
plt.title("YearPredictionMSD, SRHT sketching\n$n = 4096$, $\sigma = 10^{5}$")
plt.savefig("../plots/matrix3_error_vs_k_vs_l.pdf", format='pdf')
plt.show()
