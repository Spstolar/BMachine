import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

length_of_vecs = 100
histData = np.load('hidden_activations.npy')
my_data = histData  # Replace with data to be used.

pca = PCA(n_components=length_of_vecs)  # This creates a PCA-doing object.
pca.fit(my_data)

myPCAEigs = pca.explained_variance_  # Creates the vector of eigenvalues for the cov matrix.
plt.plot(myPCAEigs,'ro')  # Create a simple plot of the eigenvalues.
plt.title('Eigenvalues for Data')
plt.xlabel('Eigenvalue number')
plt.ylabel('Eigenvalue')
plt.savefig('eigPlot.png')  # Save the plot.

plt.clf()  # Clear this figure object for other use.

classes = np.identity(10)  # Replace with class labels matrix.

class_labels = np.zeros(classes.shape[0])  # Empty vector to condense the class label matrix.
for i in range(classes.shape[0]):
    class_labels[i] = np.argmax(classes[i,:])

num_classes = classes.shape[1]

# Generate a gradient of colors in hsv format.
hsv_colors = [(x*1.0/num_classes, x*0.5/num_classes, (num_classes-x)*0.5/num_classes) for x in range(num_classes)]
color_list = map(lambda x: colors.rgb2hex(x), hsv_colors)  # Convert the colors to hex format.

# This makes a tuple rather than a numpy array out of the labels:
label_tuple = []
for i in range(classes.shape[0]):
    label_tuple.append(int(class_labels[i]))

label_colors = np.choose(label_tuple, color_list)  # This replaces each label with the corresponding color.

fig = plt.figure(1, figsize=(4,3))
ax = fig.add_subplot(111, projection='3d')
pca = PCA(n_components=3)  # Now we force projection onto the principal components.
pca.fit(my_data)
data_proj = pca.transform(my_data)

num_pts = 10  # Let this be 100 or 1000. Too many points makes things too messy.

ax.scatter(data_proj[:num_pts,0],data_proj[:num_pts,1], data_proj[:num_pts,2], c=label_colors[:num_pts], cmap=plt.cm.spectral)

plt.axis('on')

# Remove the labels and their silly, imaginary units.
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.title('PCA Plot for Data')

plt.savefig('pca_example.png')