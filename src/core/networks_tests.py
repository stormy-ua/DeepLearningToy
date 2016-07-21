import unittest
from nodes import *
import numpy as np
from numpy.testing import *
from asserts import *
from simulation import *
from activations import *
from networks import *
from sklearn import datasets


class NeuralNetwork2LayerTest(unittest.TestCase):
    @staticmethod
    def load_mnist_data():
        # load MNIST sample data
        mnist = datasets.load_digits()
        X = np.array(mnist.data)
        y = np.array(mnist.target)
        one_hot_y = np.zeros(shape=(len(np.unique(y)), X.shape[0]))
        for i in range(0, X.shape[0]):
            one_hot_y[y[i], i] = 1
        mean, std = np.mean(X), np.std(X)
        X = (X - mean) / std
        return (X, y, one_hot_y)

    def test_backward(self):
        # load MNIST sample data
        (X, y, one_hot_y) = self.load_mnist_data()
        # do forward propagation
        ctx = SimulationContext()
        # constants
        x_in = ctx.constant(X.T)
        one_hot_y_in = ctx.constant(one_hot_y)
        (weights, biases, loss) = neural_network_2layer(ctx, 50, 10, x_in, one_hot_y_in)
        ctx.forward_backward()
        # compare calculated and numerical gradients
        w1 = weights[0]
        grad = np.zeros(shape=w1.value.shape)
        for j in [(i, k) for i in range(0, w1.value.shape[0]) for k in range(w1.value.shape[1])]:
            mask = np.zeros(shape=w1.value.shape)
            mask[j] = 1.
            grad[j] = numericalGradient(ctx, w1, loss, mask)
        assert_array_almost_equal(w1.gradient, grad, 4)
