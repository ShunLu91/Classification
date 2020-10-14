from operations import *


class Network:
    def __init__(self, batchsize, layers, dimension, dropout_rate):
        self.network = list()
        self.training = True
        for i in range(layers):
            # fc layer
            if i == 0:
                self.network.append(FC_Layer([batchsize, 28, 28, 1], dimension[i]))
            else:
                self.network.append(FC_Layer(self.network[-1].output_shape, dimension[i]))
            # relu layer
            if i != layers - 1:
                self.network.append(Relu_Layer(self.network[-1].output_shape))
                if dropout_rate > 0:
                    self.network.append(Dropout_Layer(self.network[-1].output_shape, dropout_rate))
        self.softmax = SoftMax_Layer(self.network[-1].output_shape)

    def forward(self, x):
        for i, op in enumerate(self.network):
            if 'Dropout_Layer' in str(op):
                x = op.forward(x, self.training)
            else:
                x = op.forward(x)
        return x

    def optimize(self, lr, weight_decay):
        self.softmax.gradient()
        eta = self.softmax.eta
        for op in self.network[::-1]:
            eta = op.gradient(eta)
        for op in self.network[::-1]:
            if 'FC_Layer' in str(op):
                op.backward(lr, weight_decay)
