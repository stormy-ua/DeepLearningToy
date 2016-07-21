import unittest
from nodes import *
import numpy as np
from numpy.testing import *
from asserts import *


class SumNodesTests(unittest.TestCase):
    def testForward(self):
        in1 = Connection(np.array([0., 1., 2.]))
        in2 = Connection(np.array([3., 4., 5.]))
        out = Connection()
        sum = SumNode(in1, in2, out)
        sum.forward()
        assert_array_equal(out.value, np.array([3., 5., 7.]))

    def testBackward(self):
        in1 = Connection(np.array([0., 1., 2.]))
        in2 = Connection(np.array([3., 4., 5.]))
        out = Connection(gradient=np.array([7., 8., 9.]))
        sum = SumNode(in1, in2, out)
        sum.forward()
        sum.backward()
        assert_array_almost_equal(in1.gradient, numericalGradient(sum, in1, out))
        assert_array_almost_equal(in2.gradient, numericalGradient(sum, in2, out))


class MultiplyNodesTests(unittest.TestCase):
    def testForward(self):
        in1 = Connection(np.array([0., 1., 2.]))
        in2 = Connection(np.array([3., 4., 5.]))
        out = Connection()
        operation = MultiplyNode(in1, in2, out)
        operation.forward()
        assert_array_equal(out.value, np.array([0., 4., 10.]))

    def testBackward(self):
        in1 = Connection(np.array([0., 1., 2.]))
        in2 = Connection(np.array([3., 4., 5.]))
        out = Connection(gradient=np.array([7., 8., 9.]))
        operation = MultiplyNode(in1, in2, out)
        operation.forward()
        operation.backward()
        assert_array_almost_equal(in1.gradient, numericalGradient(operation, in1, out))
        assert_array_almost_equal(in2.gradient, numericalGradient(operation, in2, out))


class DivideNodesTests(unittest.TestCase):
    def testForward(self):
        in1 = Connection(np.array([0., 1., 5.]))
        in2 = Connection(np.array([3., 4., 5.]))
        out = Connection()
        operation = DivNode(in1, in2, out)
        operation.forward()
        assert_array_equal(out.value, np.array([0., 0.25, 1.]))

    def testBackward(self):
        in1 = Connection(np.array([0., 1., 2.]))
        in2 = Connection(np.array([3., 4., 5.]))
        out = Connection(gradient=np.array([7., 8., 9.]))
        operation = DivNode(in1, in2, out)
        operation.forward()
        operation.backward()
        assert_array_almost_equal(in1.gradient, numericalGradient(operation, in1, out))
        assert_array_almost_equal(in2.gradient, numericalGradient(operation, in2, out))


class ExpNodesTests(unittest.TestCase):
    def testForward(self):
        in1 = Connection(np.array([0., 1., 5.]))
        out = Connection()
        operation = ExpNode(in1, out)
        operation.forward()
        assert_array_equal(out.value, np.array([1., np.exp(1.), np.exp(5.)]))

    def testBackward(self):
        in1 = Connection(np.array([0., 1., 2.]))
        out = Connection(gradient=np.array([7., 8., 9.]))
        operation = ExpNode(in1, out)
        operation.forward()
        operation.backward()
        assert_array_almost_equal(in1.gradient, numericalGradient(operation, in1, out))


class LogNodesTests(unittest.TestCase):
    def testForward(self):
        in1 = Connection(np.array([1., 2., 5.]))
        out = Connection()
        operation = LogNode(in1, out)
        operation.forward()
        assert_array_equal(out.value, np.array([0., np.log(2.), np.log(5.)]))

    def testBackward(self):
        in1 = Connection(np.array([0., 1., 2.]))
        out = Connection(gradient=np.array([7., 8., 9.]))
        operation = LogNode(in1, out)
        operation.forward()
        operation.backward()
        assert_array_almost_equal(in1.gradient, numericalGradient(operation, in1, out))


class ReduceSumNodesTests(unittest.TestCase):
    def testForwardFullReduce(self):
        in1 = Connection(np.array([[1., 2., 3.], [4., 5., 6.]]))
        out = Connection()
        operation = ReduceSumNode(in1, out)
        operation.forward()
        self.assertEqual(out.value, 21)

    def testForwardReduceFirstAxis(self):
        in1 = Connection(np.array([[1., 2., 3.], [4., 5., 6.]]))
        out = Connection()
        operation = ReduceSumNode(in1, out, axis=0)
        operation.forward()
        assert_array_equal(out.value, np.array([5., 7., 9.]))

    def testForwardReduceSecondAxis(self):
        in1 = Connection(np.array([[1., 2., 3.], [4., 5., 6.]]))
        out = Connection()
        operation = ReduceSumNode(in1, out, axis=1)
        operation.forward()
        assert_array_equal(out.value, np.array([6, 15]))

    def testBackwardFullReduce(self):
        in1 = Connection(np.array([[1., 2., 3.], [4., 5., 6.]]))
        out = Connection(gradient=2)
        operation = ReduceSumNode(in1, out)
        operation.forward()
        operation.backward()
        assert_array_equal(in1.gradient, np.ones(shape=in1.value.shape) * out.gradient)

    def testBackwardReduceFirstAxis(self):
        in1 = Connection(np.array([[1., 2., 3.], [4., 5., 6.]]))
        out = Connection(gradient=2)
        operation = ReduceSumNode(in1, out, axis=0)
        operation.forward()
        operation.backward()
        assert_array_equal(in1.gradient, np.ones(shape=in1.value.shape) * out.gradient)

    def testBackwardReduceSecondAxis(self):
        in1 = Connection(np.array([[1., 2., 3.], [4., 5., 6.]]))
        out = Connection(gradient=2)
        operation = ReduceSumNode(in1, out, axis=1)
        operation.forward()
        operation.backward()
        assert_array_equal(in1.gradient, np.ones(shape=in1.value.shape) * out.gradient)


class MatrixMultiplyNodesTests(unittest.TestCase):
    def testForward(self):
        in1 = Connection(np.array([[0., 1., 2.], [3., 4., 5.]]))
        in2 = Connection(np.array([6., 7., 8.]))
        out = Connection()
        operation = MatrixMultiplyNode(in1, in2, out)
        operation.forward()
        assert_array_equal(out.value, np.array([23., 86.]))

    def testBackward(self):
        in1 = Connection(np.array([[0., 1., 2.], [3., 4., 5.]]))
        in2 = Connection(np.array([6., 7., 8.])[:, np.newaxis])
        out = Connection(gradient=np.array([1., 1.])[:, np.newaxis])
        operation = MatrixMultiplyNode(in1, in2, out)
        operation.forward()
        operation.backward()
        grad = [
            [numericalGradient(operation, in1, out, np.array([[1., 0., 0.], [0., 0., 0.]]))[0, 0],
             numericalGradient(operation, in1, out, np.array([[0., 1., 0.], [0., 0., 0.]]))[0, 0],
             numericalGradient(operation, in1, out, np.array([[0., 0., 1.], [0., 0., 0.]]))[0, 0]],
            [numericalGradient(operation, in1, out, np.array([[0., 0., 0.], [1., 0., 0.]]))[1, 0],
             numericalGradient(operation, in1, out, np.array([[0., 0., 0.], [0., 1., 0.]]))[1, 0],
             numericalGradient(operation, in1, out, np.array([[0., 0., 0.], [0., 0., 1.]]))[1, 0]]
            ]
        assert_array_almost_equal(in1.gradient, np.array(grad), 5)


class MaxNodeTests(unittest.TestCase):
    def testForward(self):
        in1 = Connection(np.array([-1., 1., 0.]))
        in2 = Connection(np.array([1., 0., -5.]))
        out = Connection()
        operation = MaxNode(in1, in2, out)
        operation.forward()
        assert_array_equal(out.value, np.array([1., 1., 0.]))

    def testBackward(self):
        in1 = Connection(np.array([-1., 1., 5.]))
        in2 = Connection(np.array([1., 0., -5.]))
        out = Connection(gradient=np.array([7., 8., 9.]))
        operation = MaxNode(in1, in2, out)
        operation.forward()
        operation.backward()
        assert_array_almost_equal(in1.gradient, numericalGradient(operation, in1, out))
        assert_array_almost_equal(in2.gradient, numericalGradient(operation, in2, out))