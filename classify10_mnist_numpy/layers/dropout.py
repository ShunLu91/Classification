import numpy as np


class Dropout(object):
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
