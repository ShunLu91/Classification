import sys
sys.path.append('.')
sys.path.append('..')

import time
import numpy as np
import matplotlib.pyplot as plt

from classify10_mnist_numpy.layers.relu import Relu
from classify10_mnist_numpy.layers.fc import FullyConnect
from classify10_mnist_numpy.layers.dropout import Dropout
from classify10_mnist_numpy.layers.softmax import Softmax
from classify10_mnist_numpy.utils import load_mnist, add_noise, plot_curve


class MLP_Net:
    def __init__(self, num_layers, fc_dim, use_dp=False, dp_prob=0.2):
        assert num_layers == len(fc_dim)
        self.layers = []
        self.training = True
        for i in range(num_layers):
            # add fc
            if i == 0:
                self.layers.append(FullyConnect([batch_size, 28, 28, 1], fc_dim[i]))
            else:
                self.layers.append(FullyConnect(self.layers[-1].output_shape, fc_dim[i]))
            # add relu
            if i != num_layers - 1:
                self.layers.append(Relu(self.layers[-1].output_shape))
                if use_dp:
                    self.layers.append(Dropout(self.layers[-1].output_shape, dp_prob))
        self.sf = Softmax(self.layers[-1].output_shape)

    def forward(self, img):
        for layer in self.layers:
            try:
                if 'Dropout' in str(layer):
                    output = layer.forward(output, self.training)
                else:
                    output = layer.forward(output)
            except:
                output = layer.forward(img)
        return output

    def cal_gradient(self):
        self.sf.gradient()
        eta = self.sf.eta
        for layer in self.layers[::-1]:
            eta = layer.gradient(eta)

    def adjust_params(self):
        for layer in self.layers[::-1]:
            if 'FullyConnect' in str(layer):
                layer.backward(learning_rate, weight_decay)


def train(print_freq=100):
    model.training = True
    train_acc = 0
    train_loss = 0

    for step in range(train_images.shape[0] // batch_size):
        batch_loss = 0
        batch_acc = 0
        img = train_images[step * batch_size:(step + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = train_labels[step * batch_size:(step + 1) * batch_size]
        if noise:
            img = add_noise(img, noise_prob)

        output = model.forward(img)

        batch_loss += model.sf.cal_loss(output, np.array(label))
        train_loss += model.sf.cal_loss(output, np.array(label))
        for j in range(batch_size):
            if np.argmax(model.sf.softmax[j]) == label[j]:
                batch_acc += 1
                train_acc += 1

        model.cal_gradient()
        model.adjust_params()

        if step % print_freq == 0:
            print(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                "  epoch: %d  step: %5d  avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (
                    epoch, step, batch_acc / float(batch_size), batch_loss / batch_size, learning_rate
                )
            )
            train_batch_loss_list.append(batch_loss / batch_size)
            train_batch_acc_list.append(batch_acc / float(batch_size))
    train_loss_list.append(train_loss / train_images.shape[0])
    train_acc_list.append(train_acc / float(train_images.shape[0]))

    print(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
        "  epoch: %d  train_acc: %.4f  avg_train_loss: %.4f" % (
            epoch, train_acc / float(train_images.shape[0]), train_loss / train_images.shape[0])
    )


def evaluate(best_acc):
    model.training = False
    val_acc = 0
    val_loss = 0

    for step in range(int(test_images.shape[0] / batch_size)):
        img = test_images[step * batch_size:(step + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = test_labels[step * batch_size:(step + 1) * batch_size]

        output = model.forward(img)

        val_loss += model.sf.cal_loss(output, np.array(label))

        for j in range(batch_size):
            if np.argmax(model.sf.softmax[j]) == label[j]:
                val_acc += 1

    if best_acc < val_acc / float(test_images.shape[0]):
        best_acc = val_acc / float(test_images.shape[0])
    test_loss_list.append(val_loss / test_images.shape[0])
    test_acc_list.append(val_acc / float(test_images.shape[0]))

    print(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
        "  epoch: %d  val_acc: %.4f  avg_val_loss: %.4f best_val_acc: %.4f" % (
            epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0], best_acc)
    )

    return best_acc


if __name__ == '__main__':
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 4e-4
    best_acc = 0.0

    num_layers = 2
    fc_dim = [500, 10]

    train_batch_loss_list = []
    train_batch_acc_list = []
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    noise = False
    noise_prob = 0.9

    print('num_epochs: %s, batch_size: %s, learning_rate: %s, weight_decay: %s, '
          'num_layers: %s, fc_dim: %s, noise: %s, noise_prob: %s' %
          (num_epochs, batch_size, learning_rate, weight_decay, num_layers, fc_dim, noise, noise_prob))
    train_images, train_labels = load_mnist('./dataset/MNIST/raw')
    test_images, test_labels = load_mnist('./dataset/MNIST/raw', 't10k')


    print(
        'train_images: %d, test_images: %d, batch_size: %d, train_step: %d, test_step: %d' %
        (len(train_images), len(test_images), batch_size, len(train_images) // batch_size,
         len(test_images) // batch_size)
    )

    model = MLP_Net(num_layers=num_layers, fc_dim=fc_dim, use_dp=False, dp_prob=0.3)
    for epoch in range(num_epochs):
        train(print_freq=50)
        best_acc = evaluate(best_acc)

    print(train_batch_loss_list)
    print(train_batch_acc_list)
    print(train_loss_list)
    print(train_acc_list)
    print(test_loss_list)
    print(test_acc_list)

