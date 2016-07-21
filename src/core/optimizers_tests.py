import unittest
from nodes import *
import numpy as np
from numpy.testing import *
from asserts import *
from simulation import *
from activations import *
from networks import *
from sklearn import datasets
from optimizers import *


class SgdOptimizerTest(unittest.TestCase):
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

    def test_one_epoch(self):
        (X, y, one_hot_y) = self.load_mnist_data()
        ctx = SimulationContext()
        x_in = ctx.constant(X.T, "X.T")
        one_hot_y_in = ctx.constant(one_hot_y, "one_hot_y")
        (weights, biases, loss) = neural_network_2layer(ctx, 50, 10, x_in, one_hot_y_in)
        sgd(ctx, x_in, one_hot_y_in)

    def test_overfit(self):
        (X, y, one_hot_y) = self.load_mnist_data()
        ctx = SimulationContext()
        x_in = ctx.constant(X.T, "X.T")
        one_hot_y_in = ctx.constant(one_hot_y, "one_hot_y")
        (weights, biases, output, loss) = neural_network_2layer(ctx, 50, 10, x_in, one_hot_y_in)

        for epoch in range(0, 200):
            sgd(ctx, x_in, one_hot_y_in, learningRate=0.05)

        x_in.value = X.T
        one_hot_y_in.value = one_hot_y
        ctx.forward()
        y_pred = np.argmax(output.value, axis=0)

        accuracy = np.sum(y_pred == y)/len(y)

        self.assertAlmostEqual(loss.value, 0)