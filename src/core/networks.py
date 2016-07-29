from simulation import *
import numpy as np
from activations import *
from losses import *
from computational_graph import *


def neural_network(cg: ComputationalGraph, x_in: Connection, n0, *n):
    _layers = []

    def add_layer(inputs_count, outputs_count, input: Connection):
        index = len(_layers) + 1
        w_in = cg.variable("W{}".format(index), 0.01 * np.random.randn(outputs_count, inputs_count))
        b_in = cg.variable("b{}".format(index), 0.01 * np.ones(outputs_count))
        layer_output = relu(cg, cg.sum(cg.matrix_multiply(w_in, input), cg.broadcast(b_in, axis=1)),
                            "layer{}_output".format(index))
        _layers.append(layer_output)

        return layer_output

    add_layer(n0, n[0], x_in)

    for n1, n2 in zip(n, n[1:]):
        add_layer(n1, n2, _layers[-1])

    return _layers[-1]
