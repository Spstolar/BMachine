import numpy as np

def whacky():
    a = np.arange(3)
    b = np.arange(4)
    c = np.arange(5)
    return a, b, c

cat_a = np.zeros(3)
cat_b = np.zeros(4)
cat_c = np.zeros(5)

for i in range(4):
    a_cat_a, a_cat_b, a_cat_c = whacky()
    cat_a += a_cat_a
    cat_b += a_cat_b
    cat_c += a_cat_c

print cat_a
print cat_b
print cat_c