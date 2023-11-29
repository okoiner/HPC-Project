import numpy as np
import matplotlib.pyplot as plt

svds = np.load("svds.npy")
svdss = np.load("svdss.npy")

plt.plot(range(svds.shape[1]), svds[0]/svds[1], '-')
plt.plot(range(svdss.shape[1]), svdss[0]/svdss[1], '-')
plt.show()

#fig = plt.figure()
#ax = plt.gca()
#ax.plot(svds[0], svds[1], 'o')
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()
