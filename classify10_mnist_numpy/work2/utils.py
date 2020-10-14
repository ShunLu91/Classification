import struct
import numpy as np


def get_data(path, type='train'):
    images_path = '%s/%s-images-idx3-ubyte' % (path, type)
    labels_path = '%s/%s-labels-idx1-ubyte' % (path, type)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def random_noise(image, noise_rate):
    noise = np.random.randn(
        image.shape[0], image.shape[1], image.shape[2], image.shape[3]
    )
    noise_img = np.array(image, dtype=np.float) / 255 + noise * noise_rate
    noise_img = 255 * noise_img / np.max(noise_img)
    noise_img = np.array(np.clip(noise_img, 0, 255), dtype=np.uint8)

    return noise_img

