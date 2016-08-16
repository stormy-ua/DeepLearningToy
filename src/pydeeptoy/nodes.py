import numpy as np
from abc import ABCMeta, abstractmethod


class Connection:
    def __init__(self, name="", init_value=None):
        self._name = name
        self._init_value = init_value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def init_value(self):
        return self._init_value

    def __str__(self):
        return self._name


class ConnectionData:
    def __init__(self, value=None, gradient=None):
        self._value = value
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


class Variable(Connection):
    def __init__(self, name="", init_value=None, shape=None):
        super().__init__(name, init_value)
        self._shape = shape


class Constant(Connection):
    def __init__(self, init_value=None, name=""):
        super().__init__(name, init_value)

    def __str__(self):
        return 'constant={}'.format(self._init_value)


class Node:
    __metaclass__ = ABCMeta

    def __init__(self, inputs=[], outputs=[]):
        self.inputs = inputs
        self.outputs = outputs

    @abstractmethod
    def forward(self, data_bag):
        [data_bag[i].reset_gradient() for i in self.inputs]

    @abstractmethod
    def backward(self, data_bag): pass

    def forward_backward(self, data_bag):
        self.forward(data_bag)
        self.backward(data_bag)


class SumNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = data_bag[self.in1].value + data_bag[self.in2].value

    def backward(self, data_bag):
        data_bag[self.in1].gradient = data_bag[self.out].gradient
        data_bag[self.in2].gradient = data_bag[self.out].gradient


class MultiplyNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = data_bag[self.in1].value * data_bag[self.in2].value

    def backward(self, data_bag):
        data_bag[self.in1].gradient = data_bag[self.in2].value * data_bag[self.out].gradient
        data_bag[self.in2].gradient = data_bag[self.in1].value * data_bag[self.out].gradient


class DivNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = data_bag[self.in1].value / data_bag[self.in2].value

    def backward(self, data_bag):
        data_bag[self.in1].gradient = (1 / data_bag[self.in2].value) * data_bag[self.out].gradient
        data_bag[self.in2].gradient = data_bag[self.in1].value * (-1 / data_bag[self.in2].value ** 2) * data_bag[
            self.out].gradient


class ExpNode(Node):
    def __init__(self, in1: Connection, out: Connection):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = np.exp(data_bag[self.in1].value)

    def backward(self, data_bag):
        data_bag[self.in1].gradient = np.exp(data_bag[self.in1].value) * data_bag[self.out].gradient


class SqrtNode(Node):
    def __init__(self, in1: Connection, out: Connection):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = np.sqrt(data_bag[self.in1].value)

    def backward(self, data_bag):
        data_bag[self.in1].gradient = (.5/np.sqrt(data_bag[self.in1].value)) * data_bag[self.out].gradient


class LogNode(Node):
    def __init__(self, in1: Connection, out: Connection):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = np.log(data_bag[self.in1].value)

    def backward(self, data_bag):
        data_bag[self.in1].gradient = (1 / data_bag[self.in1].value) * data_bag[self.out].gradient


class ExpressionNode(Node):
    def __init__(self, in1: Connection, out: Connection, expression):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out
        self.expression = expression

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = self.expression(data_bag[self.in1].value)

    def backward(self, data_bag):
        pass


class ReduceSumNode(Node):
    def __init__(self, in1: Connection, out: Connection, axis=None):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out
        self.axis = axis

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = np.sum(data_bag[self.in1].value, self.axis)

    def backward(self, data_bag):
        data_bag[self.in1].gradient = np.ones(shape=data_bag[self.in1].value.shape) * data_bag[self.out].gradient


class BroadcastNode(Node):
    def __init__(self, in1: Connection, out: Connection, axis=1):
        super().__init__([in1], [out])
        if axis != 1:
            raise Exception("Axis other than 1 are not supported for now")
        self.in1 = in1
        self.out = out
        self.axis = axis

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = data_bag[self.in1].value[:, np.newaxis]

    def backward(self, data_bag):
        data_bag[self.in1].gradient = np.sum(data_bag[self.out].gradient, axis=1)


class MatrixMultiplyNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = data_bag[self.in1].value.dot(data_bag[self.in2].value)

    def backward(self, data_bag):
        data_bag[self.in1].gradient = data_bag[self.out].gradient.dot(data_bag[self.in2].value.T)
        data_bag[self.in2].gradient = data_bag[self.in1].value.T.dot(data_bag[self.out].gradient)


class MaxNode(Node):
    def __init__(self, in1: Connection, in2: Connection, out: Connection):
        super().__init__([in1, in2], [out])
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = np.maximum(data_bag[self.in1].value, data_bag[self.in2].value)

    def backward(self, data_bag):
        data_bag[self.in1].gradient = np.array(data_bag[self.in1].value > data_bag[self.in2].value, dtype=np.float32) * \
                                      data_bag[self.out].gradient
        data_bag[self.in2].gradient = np.array(data_bag[self.in2].value > data_bag[self.in1].value, dtype=np.float32) * \
                                      data_bag[self.out].gradient


class TransposeNode(Node):
    def __init__(self, in1: Connection, out: Connection, axes):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out
        self.axes = axes

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = np.transpose(data_bag[self.in1].value, self.axes)

    def backward(self, data_bag):
        data_bag[self.in1].gradient = np.transpose(data_bag[self.out].gradient, self.axes)


class ReshapeNode(Node):
    def __init__(self, in1: Connection, out: Connection, newshape):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out
        self.newshape = newshape

    def forward(self, data_bag):
        super().forward(data_bag)
        data_bag[self.out].value = np.reshape(data_bag[self.in1].value, self.newshape)

    def backward(self, data_bag):
        data_bag[self.in1].gradient = np.reshape(data_bag[self.out].gradient, data_bag[self.in1].value.shape)


class Tensor3dToCol(Node):
    def __init__(self, in1: Connection, out: Connection, receptive_field_size, stride=1, padding=1):
        super().__init__([in1], [out])
        self.in1 = in1
        self.out = out
        self.receptive_field_size = receptive_field_size
        self.stride = stride
        self.padding = padding

    @staticmethod
    def get_indices(f, c, s, output_height, output_width):
        io = np.repeat(s * np.arange(output_height, dtype=np.int32), output_height * f * f * c)
        ko = np.tile(np.repeat(s * np.arange(output_width, dtype=np.int32), f * f * c), output_width)

        i = np.tile(np.tile(np.repeat(np.arange(f, dtype=np.int32), f), c), output_height * output_width)
        k = np.tile(np.tile(np.tile(np.arange(f, dtype=np.int32), f), output_height * output_width), c)

        j = np.tile(np.repeat(np.arange(c, dtype=np.int32), f * f), output_height * output_width)

        return slice(None), j, i + io, k + ko

    @staticmethod
    def get_output_dims(w, h, f, p, s):
        assert (w - f + 2 * p) % s == 0

        output_width = (w - f + 2 * p) / s + 1
        output_height = (h - f + 2 * p) / s + 1
        return output_width, output_height

    @staticmethod
    def pad(x, p):
        return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    def forward(self, data_bag):
        super().forward(data_bag)
        x = data_bag[self.in1].value
        # input width
        w = x.shape[2]
        # input height
        h = x.shape[3]
        # number of samples
        n = x.shape[0]
        # depth
        c = x.shape[1]
        f = self.receptive_field_size
        p = self.padding
        s = self.stride

        x_padded = self.pad(x, p)

        output_width, output_height = self.get_output_dims(w, h, f, p, s)
        idx = self.get_indices(f, c, s, output_height, output_width)

        x_col = x_padded[idx].reshape(n, output_height * output_width, -1)

        data_bag[self.out].value = x_col

    def backward(self, data_bag):
        x = data_bag[self.in1].value
        # input width
        w = x.shape[2]
        # input height
        h = x.shape[3]
        # number of samples
        n = x.shape[0]
        # depth
        c = x.shape[1]
        f = self.receptive_field_size
        p = self.padding
        s = self.stride

        grad_in = data_bag[self.out].gradient

        output_width, output_height = self.get_output_dims(w, h, f, p, s)
        idx = self.get_indices(f, c, s, output_height, output_width)

        grad = self.pad(np.zeros(shape=x.shape), p)
        np.add.at(grad, idx, grad_in.reshape(n, -1))

        if p != 0:
            grad = grad[:, :, p:-p, p:-p]

        data_bag[self.in1].gradient = grad
