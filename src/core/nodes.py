import numpy as np
from abc import ABCMeta, abstractmethod


class Connection:
    _value = None
    _gradient = None

    def __init__(self, value=None, gradient=None):
        self.value = value
        self._gradient = gradient

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, value):
        if self.gradient is None or value is None:
            self._gradient = value
        else:
            self._gradient += value

    def reset_gradient(self, to_value=None):
        self._gradient = to_value


class Node:
    __metaclass__ = ABCMeta

    def __init__(self, inputs=[], outputs=[]):
        self.inputs = inputs
        self.outputs = outputs

    @abstractmethod
    def forward(self):
        [i.reset_gradient() for i in self.inputs]

    @abstractmethod
    def backward(self): pass


class SumNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self):
        super().forward()
        self.out.value = self.in1.value + self.in2.value

    def backward(self):
        self.in1.gradient = self.out.gradient
        self.in2.gradient = self.out.gradient


class MultiplyNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self):
        super().forward()
        self.out.value = self.in1.value * self.in2.value

    def backward(self):
        self.in1.gradient = self.in2.value * self.out.gradient
        self.in2.gradient = self.in1.value * self.out.gradient


class DivNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self):
        super().forward()
        self.out.value = self.in1.value / self.in2.value

    def backward(self):
        self.in1.gradient = (1 / self.in2.value) * self.out.gradient
        self.in2.gradient = self.in1.value * (-1 / self.in2.value ** 2) * self.out.gradient


class ExpNode(Node):
    def __init__(self, in1: Connection, out: Connection):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out

    def forward(self):
        super().forward()
        self.out.value = np.exp(self.in1.value)

    def backward(self):
        self.in1.gradient = np.exp(self.in1.value) * self.out.gradient


class LogNode(Node):
    def __init__(self, in1: Connection, out: Connection):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out

    def forward(self):
        super().forward()
        self.out.value = np.log(self.in1.value)

    def backward(self):
        self.in1.gradient = (1 / self.in1.value) * self.out.gradient


class ReduceSumNode(Node):
    def __init__(self, in1: Connection, out: Connection, axis=None):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out
        self.axis = axis

    def forward(self):
        super().forward()
        self.out.value = np.sum(self.in1.value, self.axis)

    def backward(self):
        self.in1.gradient = np.ones(shape=self.in1.value.shape) * self.out.gradient


class MatrixMultiplyNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self):
        super().forward()
        self.out.value = self.in1.value.dot(self.in2.value)

    def backward(self):
        self.in1.gradient = self.out.gradient.dot(self.in2.value.T)
        self.in2.gradient = self.in1.value.T.dot(self.out.gradient)


class MaxNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self):
        super().forward()
        self.out.value = np.maximum(self.in1.value, self.in2.value)

    def backward(self):
        self.in1.gradient = np.array(self.in1.value > self.in2.value, dtype=np.float32) * self.out.gradient
        self.in2.gradient = np.array(self.in2.value > self.in1.value, dtype=np.float32) * self.out.gradient

class Node2:
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, x, *args): pass

    @abstractmethod
    def backward(self, dy): pass


class ReLuNode2(Node2):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dy):
        grad = np.array(self.x > 0, dtype=np.float32) * dy
        return grad


class SigmoidNode(Node2):
    def forward(self, x):
        self.x = x
        self.sigmoid = 1 / (1 + np.exp(-1 * x))
        return self.sigmoid

    def backward(self, dy):
        grad = self.sigmoid * (1 - self.sigmoid) * dy
        return grad


class DropoutNode(Node2):
    p = 0.5

    def forward(self, x):
        self.x = x
        self.f1 = np.random.rand(*x.shape) < self.p
        self.dropout = x * self.f1
        return self.dropout

    def backward(self):
        grad = self.f1
        return grad


class SoftmaxLossNode(Node2):
    def forward(self, x, vectY):
        self.x = x
        self.vectY = vectY
        self.f1 = np.exp(x)
        self.f2 = self.f1 * vectY
        self.f3 = np.sum(self.f2, axis=0)
        self.f4 = np.sum(self.f1, axis=0)
        self.f5 = self.f3 / self.f4
        self.f6 = -np.log(self.f5)
        self.f7 = np.sum(self.f6)
        self.softmaxLoss = self.f7 / x.shape[1]
        return self.softmaxLoss

    def backward(self):
        # 1
        df6 = np.ones(shape=self.f6.shape)
        # 2
        df5 = -df6 / self.f5
        # 3
        df4 = -self.f3 / (self.f4 ** 2) * df5
        # 2
        df1 = np.ones(shape=self.f1.shape) * df4
        df3 = df5 / self.f4
        self.df2 = np.ones(shape=self.f2.shape) * df3
        df1 += self.vectY * self.df2
        # 1
        grad = self.f1 * df1 / self.x.shape[1]
        return grad


class TanhNode(Node2):
    def forward(self, x):
        self.x = x
        self.tanh = np.tanh(x)
        return self.tanh

    def backward(self):
        grad = 1 - self.tanh ** 2
        return grad


class MatrixMulNode(Node2):
    def forward(self, X, Y):
        self.Y = Y
        self.X = X
        mul = X.dot(Y)
        return mul

    def backward(self, dy):
        grad = (dy.dot(self.Y.T), self.X.T.dot(dy))
        return grad


class HingeLossNode(Node2):
    def forward(self, N, x, vectY):
        self.N = N
        self.vectY = vectY
        # 2
        self.f2 = x * vectY
        # 3
        self.f3 = np.sum(self.f2, axis=0)
        # 4
        self.f4 = x - self.f3 + 1
        # 5
        self.f5 = self.f4 * (1 - vectY)
        # 6
        self.f6 = np.maximum(self.f5, 0)
        # 7
        self.f7 = np.sum(self.f6)
        # 8
        self.f8 = self.f7 / N

        loss = np.sum(self.f8)
        return loss

    def backward(self):
        # 8
        df7 = 1 / self.N
        # 7
        # df7 = np.ones(shape = self.f7.shape)
        df6 = np.ones(shape=self.f6.shape) * df7
        # 6
        # df6 = np.ones(shape = self.f6.shape)
        df5 = np.array(self.f5 > 0, dtype=np.float32) * df6
        # 5
        # df5 = np.ones(shape = self.f5.shape)
        df4 = df5 * (1 - self.vectY)
        # 4
        # df4 = np.ones(shape = self.f4.shape)
        df3 = -1 * np.ones(shape=self.f3.shape) * np.sum(df4, axis=0)
        # 3+2
        df1 = df4
        df2 = df3
        df1 += df2 * self.vectY

        grad = df1
        return grad
