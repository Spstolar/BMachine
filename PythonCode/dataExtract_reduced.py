import numpy as np

# url to format of files: http://yann.lecun.com/exdb/mnist/
num_train_examples = 60000
num_test_examples = 10000
num_classes = 10
pixels_per_image = 784
bits_per_pixel = 8
header_bytes_images = 16
header_bytes_labels = 8

def bits(f):
    bytes = (ord(b) for b in f.read())  # stores all bytes in the file
    for b in bytes:
        '''
        The following line only returns the last bit of every byte
        The yield command returns a generator as well explained here:
        http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
        '''
        yield (b >> 7) & 1  # I am not quite sure what the & 1 is for


def extract_images_bin2py(num_examples, header_bytes, filename, savefile, pixels_per_image=784):
    dataset = np.zeros((num_examples, pixels_per_image))
    #train_labels = np.zeros((num_train_examples, num_classes))
    #test_set = np.zeros((num_test_examples, pixels_per_pic))
    #test_labels = np.zeros((num_test_examples, num_classes))

    byte_count = 1

    for b in bits(open(filename, 'r')):

        if byte_count <= header_bytes:  # Skip header bytes
            pass

        else:
            corrected_byte_count = byte_count - header_bytes - 1
            image = corrected_byte_count / pixels_per_image  # Which example this belongs to
            pixel = corrected_byte_count % pixels_per_image  # Which pixel this belongs to
            dataset[image, pixel] = b

        byte_count = byte_count + 1
    np.save(savefile, dataset)

# arguments for training images
# filename='train-images-binary'
# savefile= 'trainSet.npy'

#arguments for test images
filename='test-images-binary'
savefile= 'testSet.npy'

extract_images_bin2py(num_test_examples, header_bytes_images, filename, savefile)
# just to remember a quick way of visualizing our data
'''
ipython
Python 2.7.13 (default, Dec 17 2016, 23:03:43) 
Type "copyright", "credits" or "license" for more information.

IPython 5.3.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: %matplotlib
Using matplotlib backend: MacOSX

In [2]: import matplotlib.pyplot as plt

In [3]: import matplotlib.image as mpimg

In [4]: import numpy as np

In [5]: img=np.random.randint(0,2,size=(28,28
   ...: ))

In [6]: imgplot=plt.imshow(img
   ...: )

In [7]: img
'''