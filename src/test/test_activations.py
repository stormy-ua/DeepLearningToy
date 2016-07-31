import unittest
from pydeeptoy.nodes import *
import numpy as np
from numpy.testing import *
from pydeeptoy.asserts import *
from pydeeptoy.simulation import *
from pydeeptoy.activations import *


class ReLuTests(unittest.TestCase):
    def test_forward_scalar(self):
        cg = ComputationalGraph()
        in1 = cg.variable()
        relu1 = relu(cg, in1)
        ctx = SimulationContext()
        ctx.forward(cg, {in1: np.array([1., -1., 0.])})
        assert_array_equal(ctx[relu1].value, np.array([1., 0., 0]))

    def test_backward_scalar(self):
        cg = ComputationalGraph()
        in1 = cg.variable()
        relu1 = relu(cg, in1)
        ctx = SimulationContext()
        ctx.forward_backward(cg, {in1: np.array([1., -1., 0.])})
        grad = [
            numerical_gradient(lambda : ctx.forward_backward(cg), ctx[in1], ctx[relu1], np.array([1., 0., 0.]))[0],
            numerical_gradient(lambda : ctx.forward_backward(cg), ctx[in1], ctx[relu1], np.array([0., 1., 0.]))[1],
            numerical_gradient(lambda : ctx.forward_backward(cg), ctx[in1], ctx[relu1], np.array([0., 0., 1.]))[2]
        ]
        assert_array_almost_equal(ctx[in1].gradient, grad)
