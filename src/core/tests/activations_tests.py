import unittest
from nodes import *
import numpy as np
from numpy.testing import *
from asserts import *
from simulation import *
from activations import *

class ReLuTests(unittest.TestCase):
    def testForwardScalar(self):
        ctx = SimulationContext()
        relu1 = relu(ctx, ctx.variable(np.array([1., -1., 0.])))
        ctx.forward()
        assert_array_equal(relu1.value, np.array([1., 0., 0]))

    def testBackwardScalar(self):
        ctx = SimulationContext()
        in1 = ctx.variable(np.array([1., -1., 0.]))
        relu1 = relu(ctx, in1)
        ctx.forward()
        ctx.backward()
        numerical_gradient = [
            numericalGradient(ctx, in1, relu1, np.array([1., 0., 0.]))[0],
            numericalGradient(ctx, in1, relu1, np.array([0., 1., 0.]))[1],
            numericalGradient(ctx, in1, relu1, np.array([0., 0., 1.]))[2]
        ]
        assert_array_almost_equal(in1.gradient, numerical_gradient)