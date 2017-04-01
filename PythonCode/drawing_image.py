import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = np.random.randint(0,2,size=(28,28))
imgplot = plt.imshow(img)
img
plt.savefig('example.png')