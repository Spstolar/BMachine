import numpy as np


header_length = 64
num_examples = 60000
pixels_per_pic = 784
bits_per_label = 8
bits_per_pixel = 8
pic_length = pixels_per_pic * bits_per_pixel

test_label_binary = np.zeros((num_examples, bits_per_label))
test_label_classes = np.zeros((num_examples, 10))

def bits(f):
    bytes = (ord(b) for b in f.read())
    for b in bytes:
        for i in xrange(8):
            yield (b >> i) & 1

i = 1
pic_count = 0

for b in bits(open('train-labels-binary', 'r')):
    if i <= header_length:   # Preamble
        pass
    else:
        example = (i - header_length - 1) / bits_per_label  # Which example this belongs to.
        bit = ((i - header_length - 1) % bits_per_label)  # Which column/bit
        test_label_binary[example, bit] = b

    i += 1

for j in range(0,num_examples):
    class_label = 0
    for k in range(0,8):
        class_label += test_label_binary[j,k] * (2 ** k)
    test_label_classes[j,class_label] = 1  # May want to change this to class_label.asType(int) or something similar.


np.save("train_labels.npy", test_label_classes)