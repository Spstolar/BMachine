import numpy as np
import matplotlib.pyplot as plt

x = np.load('energy_hist_p1.npy')

n, bins, patches = plt.hist(x, 31, normed=0, histtype='bar',facecolor='green', alpha=1, edgecolor='black')

plt.xlabel('Energy Value')
plt.ylabel('Number of Sweeps')
plt.title('Energy Values for Post-Stabilization Sweeps')
plt.axis([30,70,0,30])
plt.grid(True)
plt.grid(color='b', linestyle='-', linewidth=.1,alpha=.3)

# plt.show()  # If you don't want to save it, but just view it, use this line.
plt.savefig('example_hist.png')