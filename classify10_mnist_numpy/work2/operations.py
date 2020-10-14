import numpy as np
from functools import reduce


class FC_Layer(object):
    def __init__(self, shape, output_num=2):
        self.input_shape = shape
        self.batchsize = shape[0]

        input_len = int(reduce(lambda x, y: x * y, shape[1:]))

        self.weights = np.random.standard_normal((input_len, output_num))/100
        self.bias = np.random.standard_normal(output_num)/100

        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x = x.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.weights)+self.bias
        return output

    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)

        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


class Relu_Layer(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x < 0] = 0
        return self.eta


class SoftMax_Layer(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.batchsize = shape[0]

    def cal_loss(self, prediction, label):
        self.label = label
        self.prediction = prediction
        self.predict(prediction)
        self.loss = 0
        for i in range(self.batchsize):
            self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]

        return self.loss

    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        for i in range(self.batchsize):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax

    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.batchsize):
            self.eta[i, self.label[i]] -= 1
        return self.eta


class Dropout_Layer(object):
    def __init__(self, shape, p=0.2):
        self.p = p
        self._mask = None
        self.output_shape = shape

    def forward(self, X, training=True):
        c = (1 - self.p)
        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def gradient(self, accum_grad):
        return accum_grad * self._mask


