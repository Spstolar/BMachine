import numpy as np

num_examples = 1000

ones_vec = np.ones(5)
neg_ones_vec = -np.ones(5)

vec_1 = np.hstack((ones_vec, neg_ones_vec))
vec_2 = np.hstack((ones_vec, ones_vec))
vec_3 = np.hstack((neg_ones_vec, ones_vec))
vec_4 = np.hstack((neg_ones_vec, neg_ones_vec))

first = np.vstack((vec_1, vec_2))
second = np.vstack((vec_3, vec_4))
vecs = np.vstack((first, second))

examples = np.zeros((num_examples, 10))

for i in range(4):
    examples[i*250:(i+1)*250, :] = np.tile(vecs[i, :],(250, 1))

np.save('toy_example_set.npy', examples)
