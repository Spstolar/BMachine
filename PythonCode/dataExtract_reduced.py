import numpy as np

# url to format of files: http://yann.lecun.com/exdb/mnist/
# definition of constants corresponding to the MNIST data sets
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
filename = 'test-images-binary'
savefile = 'testSet.npy'

extract_images_bin2py(num_test_examples, header_bytes_images, filename, savefile)