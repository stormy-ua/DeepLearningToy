import numpy as np
from abc import ABCMeta, abstractmethod


class Connection:
    _value = None
    _gradient = None
    _name = ""

    def __init__(self, value=None, gradient=None, name=""):
        self._value = value
        self._gradient = gradient
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

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

    def __str__(self):
        return self._name


class Variable(Connection):
    def __init__(self, value=None, gradient=None, name=""):
        super().__init__(value, gradient, name)


class Constant(Connection):
    def __init__(self, value=None, gradient=None, name=""):
        super().__init__(value, gradient, name)


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


class BroadcastNode(Node):
    def __init__(self, in1: Connection, out: Connection, axis=1):
        super().__init__([in1], [out])
        if axis != 1:
            raise Exception("Axis other than 1 are not supported for now")
        self.in1 = in1
        self.out = out
        self.axis = axis

    def forward(self):
        super().forward()
        self.out.value = self.in1.value[:, np.newaxis]

    def backward(self):
        self.in1.gradient = np.sum(self.out.gradient, axis=1)


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


