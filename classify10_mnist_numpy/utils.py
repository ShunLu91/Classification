import struct
import numpy as np


def load_mnist(path, type='train'):
    images_path = '%s/%s-images-idx3-ubyte' % (path, type)
    labels_path = '%s/%s-labels-idx1-ubyte' % (path, type)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def add_noise(matrix, noise_prob):
    """
    Input
        matrix: the original image
    Return
        noise_img: matrix + noise
    """
    mask = np.random.uniform(size=matrix.shape) > noise_prob
    noise_img = matrix * mask

    return noise_img

