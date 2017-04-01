import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

length = 28

dog = np.load('testSetSimple.npy')

for j in range(10):
    img = dog[j,:].reshape((28,28))
    # img = np.random.randint(0,2,size=(length, length))
    imgplot = plt.imshow(img)
    img
    plt.savefig('example' + str(j) + '.png')