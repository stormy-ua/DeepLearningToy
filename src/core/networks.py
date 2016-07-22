from simulation import *
import numpy as np
from activations import *
from losses import *


class NeuralNetwork:
    _layers = []

    @property
    def output(self):
        return self._layers[-1]

    def __init__(self, n0, *n):
        if len(n) == 0:
            raise Exception("Network must have at least one layer")

        self.ctx = SimulationContext()
        self.x_in = self.ctx.constant(name="X.T")
        self.one_hot_y_in = self.ctx.constant(name="one_hot_y")
        self.weights = []
        self.biases = []

        self.add_layer(n0, n[0], self.x_in)

        for n1, n2 in zip(n, n[1:]):
            self.add_layer(n1, n2, self.output)

        # loss
        self.loss = softmax(self.ctx, self.output, self.one_hot_y_in, n0, "loss_softmax")

    def add_layer(self, inputs_count, outputs_count, input: Connection):
        index = len(self._layers) + 1
        W = 0.01 * np.random.randn(outputs_count, inputs_count)
        b = 0.01 * np.ones(outputs_count)
        w_in = self.ctx.variable(W, "W{}".format(index))
        b_in = self.ctx.variable(b, "b{}".format(index))
        layer_output = relu(self.ctx,
                             self.ctx.sum(self.ctx.matrix_multiply(w_in, input), self.ctx.broadcast(b_in, axis=1)),
                             "layer{}_output".format(index))
        self.weights.append(w_in)
        self.biases.append(b_in)
        self._layers.append(layer_output)

        return layer_output

    def forward_backward(self, params=dict()):
        self.ctx.forward_backward(params)

    def predict(self, X):
        self.x_in.value = X.T
        self.ctx.forward()
        return self.output.value

