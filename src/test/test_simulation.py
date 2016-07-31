from pydeeptoy.asserts import *
from pydeeptoy.losses import *
from pydeeptoy.simulation import *
import unittest
from numpy.testing import *


class SimulationTests(unittest.TestCase):
    def test_forward_scalar(self):
        cg = ComputationalGraph()
        in1 = cg.variable()
        in2 = cg.variable()
        div1 = cg.div(cg.multiply(cg.sum(in1, in2), cg.constant(3)), cg.constant(9))
        ctx = SimulationContext()
        ctx.forward(cg, {in1: 1, in2: 2})
        self.assertEqual(ctx[div1].value, 1)

    def test_backward_scalar(self):
        context = SimulationContext()
        cg = ComputationalGraph()
        in1 = cg.variable()
        in2 = cg.variable()
        cg.div(cg.multiply(cg.sum(in1, in2), cg.constant(4)), cg.constant(8))
        context.forward_backward(cg, {in1: 1, in2: 2})
        self.assertEqual(context[in1].gradient, .5)

    def test_forward_softmax(self):
        cg = ComputationalGraph()
        x = cg.variable()
        one_hot_y = cg.variable()
        cost = softmax(cg, x, one_hot_y)
        ctx = SimulationContext()
        ctx.forward(cg, {x: np.array([-2.85, 0.86, 0.28])[:, np.newaxis],
                                         one_hot_y: np.array([0., 0., 1.])[:, np.newaxis]})
        assert_array_almost_equal(ctx[cost].value, np.array([1.04]), 3)

    def test_backward_softmax(self):
        cg = ComputationalGraph()
        x = cg.variable()
        one_hot_y = cg.variable()
        cost = softmax(cg, x, one_hot_y)
        ctx = SimulationContext()
        ctx.forward_backward(cg, {x: np.array([-2.85, 0.86, 0.28])[:, np.newaxis],
                                  one_hot_y: np.array([0., 0., 1.])[:, np.newaxis]})
        gradient = [
            numerical_gradient(lambda: ctx.forward_backward(cg), ctx[x], ctx[cost], np.array([[1.], [0.], [0.]])),
            numerical_gradient(lambda: ctx.forward_backward(cg), ctx[x], ctx[cost], np.array([[0.], [1.], [0.]])),
            numerical_gradient(lambda: ctx.forward_backward(cg), ctx[x], ctx[cost], np.array([[0.], [0.], [1.]]))
        ]
        assert_array_almost_equal(np.ravel(ctx[x].gradient), gradient)
