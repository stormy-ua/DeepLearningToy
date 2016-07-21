import unittest
from nodes import *
import numpy as np
from numpy.testing import *
from simulation import *
from asserts import *
import losses


class SimulationTests(unittest.TestCase):
    def testForwardScalar(self):
        context = SimulationContext()
        in1 = Connection(1)
        in2 = Connection(2)
        sum1 = context.sum(in1, in2)
        mul1 = context.multiply(sum1, Connection(3))
        div1 = context.div(mul1, Connection(9))
        context.forward()
        self.assertEqual(div1.value, 1)

    def testBackwardScalar(self):
        context = SimulationContext()
        in1 = Connection(1)
        in2 = Connection(2)
        sum1 = context.sum(in1, in2)
        mul1 = context.multiply(sum1, Connection(4))
        div1 = context.div(mul1, Connection(8))
        context.forward()
        context.backward()
        self.assertEqual(in1.gradient, .5)

    def testForwardSoftmax(self):
        ctx = SimulationContext()
        x = Connection(np.array([-2.85, 0.86, 0.28])[:, np.newaxis])
        one_hot_y = Connection(np.array([0., 0., 1.])[:, np.newaxis])
        cost = losses.softmax(ctx, x, one_hot_y)
        ctx.forward()
        assert_array_almost_equal(cost.value, np.array([1.04]), 3)

    def testBackwardSoftmax(self):
        ctx = SimulationContext()
        x = Connection(np.array([-2.85, 0.86, 0.28]))
        one_hot_y = Connection(np.array([0., 0., 1.]))
        cost = losses.softmax(ctx, x, one_hot_y)
        ctx.forward()
        ctx.backward()
        numerical_gradient = [
            numericalGradient(ctx, x, cost, np.array([1., 0., 0.])),
            numericalGradient(ctx, x, cost, np.array([0., 1., 0.])),
            numericalGradient(ctx, x, cost, np.array([0., 0., 1.]))
        ]
        assert_array_almost_equal(x.gradient, numerical_gradient)
