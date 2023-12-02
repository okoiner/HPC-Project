import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "../testing/" + "C_matrix3a_error_vs_k_vs_lratio.csv"
table = pd.read_csv(file_path)

colors = ["red", "blue", "green", "magenta", "purple", "gray", "red", "blue", "green", "magenta", "purple", "gray", "red", "blue", "green", "magenta", "purple", "gray","red", "blue", "green", "magenta", "purple", "gray"]

kk = np.sort(list(set(table['k'])))
for i, k in enumerate(kk):
	for sketch in range(2):
		rows = table[(table['k'] == k) & (table['sketch_matrix'] == sketch)].sort_values(by='k')
		errors = rows["error_nuc"]
		ratios = rows["ratio_lk"]
		plt.semilogy(ratios, errors, ("--+" if sketch else "-x"), c = colors[i], label= "$k = %d$, " % k + ("short-axis" if sketch else "SRHT"))
plt.xlabel("Approximation (k)")
plt.ylabel("Trace relative error")
#plt.legend()
plt.title("YearPredictionMSD, SRHT sketching\n$n = 4096$, $\sigma = 10^{5}$")
#plt.savefig("../plots/matrix3_error_vs_k_vs_l.pdf", format='pdf')
plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "../testing/" + "C_matrix3a_error_vs_k_vs_lratio.csv"
table = pd.read_csv(file_path)

colors = ["red", "blue", "green", "magenta", "purple", "gray"]

ratios = np.sort(list(set(table['ratio_lk'])))
ratioString = ["", "1.25", "1.5", "2", "3"]
for i, ratio in enumerate(ratios):
	for sketch in range(2):
		rows = table[(table['ratio_lk'] == ratio) & (table['sketch_matrix'] == sketch)].sort_values(by='k')
		errors = rows["error_nuc"]
		kk = rows["k"]
		print(".")
		plt.semilogy(kk, errors, ("--+" if sketch else "-x"), c = colors[i], label= "$\ell = %sk$, " % ratioString[i] + ("short-axis" if sketch else "SRHT"))
plt.xlabel("Approximation (k)")
plt.ylabel("Trace relative error")
plt.legend()
plt.title("YearPredictionMSD, SRHT sketching\n$n = 4096$, $\sigma = 10^{5}$")
#plt.savefig("../plots/matrix3_error_vs_k_vs_l.pdf", format='pdf')
plt.show()
"""
