import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

length = 28

dog = np.load('stabilization_small.npy')

num_plot = 0

if num_plot == 1:
    for j in range(10):
        img = dog[j,:].reshape((length, length))  # for printing numbers
        # img = np.random.randint(0,2,size=(length, length))
        imgplot = plt.imshow(img)
        plt.savefig('example' + str(j) + '.png')
else:
    small_plot = np.load('stabilization_small.npy')
    img = small_plot[:112,:].T
    imgplot = plt.imshow(img)
    plt.title('Node Stabilization for Small Weights')
    plt.xlabel('Sweep Number')
    plt.ylabel('Node Number')
    plt.savefig('stabilization_small.png')

    large_plot = np.load('stabilization_large.npy')
    img = large_plot[:112, :].T
    imgplot = plt.imshow(img)
    plt.title('Node Stabilization for Large Weights')
    plt.xlabel('Sweep Number')
    plt.ylabel('Node Number')
    plt.savefig('stabilization_large.png')