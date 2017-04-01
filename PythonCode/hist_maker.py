import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(1000)

n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Hey')
plt.ylabel('There')
plt.title(r'$\int_{\Omega} f(\omega) d\omega$')
plt.axis([40,160,0,0.03])
plt.grid(True)

plt.show()

plt.savefig('example_hist.png')