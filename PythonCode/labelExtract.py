import numpy as np


header_length = 64
num_examples = 10000
pixels_per_pic = 784
bits_per_label = 10
bits_per_pixel = 8
pic_length = pixels_per_pic * bits_per_pixel

test_label = np.zeros((num_examples, bits_per_pixel))

def bits(f):
    bytes = (ord(b) for b in f.read())
    for b in bytes:
        for i in xrange(8):
            yield (b >> i) & 1

i = 1
pic_count = 0

for b in bits(open('test-labels-binary', 'r')):
    if i <= header_length:   # Preamble
        pass
    else:
        image = (i - header_length - 1) / bits_per_label  # Which example this belongs to.
        bit = ((i - header_length - 1) % bits_per_label)  # Which column/bit
        test_label[image, bit] = b

    i += 1

np.save("test_labels.npy", test_label)