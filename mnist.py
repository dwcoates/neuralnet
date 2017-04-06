import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """
    Load MNIST data from `path`
    """
    labels_path = os.path.join(path, "{}-labels-idx1-ubyte".format(kind))
    images_path = os.path.join(path, "{}-images-idx3-ubyte".format(kind))

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, columns = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_mnist('data', kind='train')
X_test, y_test = load_mnist('data', kind='t10k')

np.savetxt('./data/train_img.csv', X_train, fmt='%i', delimiter=',')
np.savetxt('./data/train_labels.csv', y_train, fmt='%i', delimiter=',')
np.savetxt('./data/test_img.csv', X_test, fmt='%i', delimiter=',')
np.savetxt('./data/test_labels.csv', y_test, fmt='%i', delimiter=',')

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True)

ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

X_train = np.genfromtxt('./data/train_img.csv', dtype=int, delimiter=',')
y_train = np.genfromtxt('./data/train_labels.csv', dtype=int, delimiter=',')
X_test = np.genfromtxt('./data/test_img.csv', dtype=int, delimiter=',')
y_train = np.genfromtxt('./data/test_labels.csv', dtype=int, delimiter=',')
print "done."
