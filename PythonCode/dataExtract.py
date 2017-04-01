import numpy as np


num_test_examples = 60000
pixels_per_pic = 784
bits_per_pixel = 8
pic_length = pixels_per_pic * bits_per_pixel

test_set = np.zeros((num_test_examples, pic_length))

def bits(f):
    bytes = (ord(b) for b in f.read())
    for b in bytes:
        for i in xrange(8):
            yield (b >> i) & 1

i = 1
pic_count = 0

for b in bits(open('train-images-binary', 'r')):
    if i <= 128:   # Preamble
        print b,
        if (i % 32) == 0:
            print "\n"
    elif i <= pic_length + 128:
        image = (i - 128 - 1) / pic_length  # Which example this belongs to.
        bit = ((i - 128 - 1) % pic_length)  # Which column/bit
        test_set[image, bit] = b


    i = i + 1

np.save("trainSet.npy", test_set)