import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

length = 28

img = np.random.randint(0,2,size=(length, length))
imgplot = plt.imshow(img)
img
plt.savefig('example.png')