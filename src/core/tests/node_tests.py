import unittest
from nodes import *
import numpy as np
from numpy.testing import *
from asserts import *


class SumNodesTests(unittest.TestCase):
    def test_forward(self):
        in1 = Connection(name="x")
        in2 = Connection(name="y")
        out = Connection(name="sum")
        sum = SumNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([0., 1., 2.])),
                   in2: ConnectionData(value=np.array([3., 4., 5.])),
                   out: ConnectionData()}
        sum.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([3., 5., 7.]))

    def test_backward(self):
        in1 = Connection(name="x")
        in2 = Connection(name="y")
        out = Connection(name="sum")
        sum = SumNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([0., 1., 2.])),
                   in2: ConnectionData(value=np.array([3., 4., 5.])),
                   out: ConnectionData(gradient=np.array([7., 8., 9.]))}
        sum.forward_backward(dataBag)
        assert_array_almost_equal(dataBag[in1].gradient,
                                  numerical_gradient(lambda: sum.forward_backward(dataBag), dataBag[in1], dataBag[out]))
        assert_array_almost_equal(dataBag[in2].gradient,
                                  numerical_gradient(lambda: sum.forward_backward(dataBag), dataBag[in2], dataBag[out]))


class MultiplyNodesTests(unittest.TestCase):
    def test_forward(self):
        in1 = Connection()
        in2 = Connection()
        out = Connection()
        operation = MultiplyNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([0., 1., 2.])),
                   in2: ConnectionData(value=np.array([3., 4., 5.])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([0., 4., 10.]))

    def test_backward(self):
        in1 = Connection()
        in2 = Connection()
        out = Connection()
        operation = MultiplyNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([0., 1., 2.])),
                   in2: ConnectionData(value=np.array([3., 4., 5.])),
                   out: ConnectionData(gradient=np.array([7., 8., 9.]))}
        operation.forward_backward(dataBag)
        assert_array_almost_equal(dataBag[in1].gradient,
                                  numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out]))
        assert_array_almost_equal(dataBag[in2].gradient,
                                  numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in2], dataBag[out]))


class DivideNodesTests(unittest.TestCase):
    def test_forward(self):
        in1 = Connection()
        in2 = Connection()
        out = Connection()
        operation = DivNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([0., 1., 5.])),
                   in2: ConnectionData(value=np.array([3., 4., 5.])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([0., 0.25, 1.]))

    def test_backward(self):
        in1 = Connection()
        in2 = Connection()
        out = Connection()
        operation = DivNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([0., 1., 2.])),
                   in2: ConnectionData(value=np.array([3., 4., 5.])),
                   out: ConnectionData(gradient=np.array([7., 8., 9.]))}
        operation.forward_backward(dataBag)
        assert_array_almost_equal(dataBag[in1].gradient,
                                  numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out]))
        assert_array_almost_equal(dataBag[in2].gradient,
                                  numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in2], dataBag[out]))


class ExpNodesTests(unittest.TestCase):
    def test_forward(self):
        in1 = Connection()
        out = Connection()
        operation = ExpNode(in1, out)
        dataBag = {in1: ConnectionData(value=np.array([0., 1., 5.])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([1., np.exp(1.), np.exp(5.)]))

    def test_backward(self):
        in1 = Connection()
        out = Connection()
        dataBag = {in1: ConnectionData(value=np.array([0., 1., 2.])),
                   out: ConnectionData(gradient=np.array([7., 8., 9.]))}
        operation = ExpNode(in1, out)
        operation.forward_backward(dataBag)
        assert_array_almost_equal(dataBag[in1].gradient,
                                  numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out]))


class LogNodesTests(unittest.TestCase):
    def test_forward(self):
        in1 = Connection()
        out = Connection()
        operation = LogNode(in1, out)
        dataBag = {in1: ConnectionData(value=np.array([1., 2., 5.])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([0., np.log(2.), np.log(5.)]))

    def test_backward(self):
        in1 = Connection()
        out = Connection()
        operation = LogNode(in1, out)
        dataBag = {in1: ConnectionData(value=np.array([1., 2., 5.])),
                   out: ConnectionData(gradient=np.array([7., 8., 9.]))}
        operation.forward_backward(dataBag)
        assert_array_almost_equal(dataBag[in1].gradient,
                                  numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out]))


class ReduceSumNodesTests(unittest.TestCase):
    def test_forward_full_reduce(self):
        in1 = Connection()
        out = Connection()
        operation = ReduceSumNode(in1, out)
        dataBag = {in1: ConnectionData(value=np.array([[1., 2., 3.], [4., 5., 6.]])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        self.assertEqual(dataBag[out].value, 21)

    def test_forward_reduce_first_axis(self):
        in1 = Connection()
        out = Connection()
        operation = ReduceSumNode(in1, out, axis=0)
        dataBag = {in1: ConnectionData(value=np.array([[1., 2., 3.], [4., 5., 6.]])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([5., 7., 9.]))

    def test_forward_reduce_second_axis(self):
        in1 = Connection()
        out = Connection()
        operation = ReduceSumNode(in1, out, axis=1)
        dataBag = {in1: ConnectionData(value=np.array([[1., 2., 3.], [4., 5., 6.]])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([6, 15]))

    def test_backward_full_feduce(self):
        in1 = Connection()
        out = Connection()
        operation = ReduceSumNode(in1, out)
        dataBag = {in1: ConnectionData(value=np.array([[1., 2., 3.], [4., 5., 6.]])),
                   out: ConnectionData(gradient=2)}
        operation.forward_backward(dataBag)
        assert_array_equal(dataBag[in1].gradient, np.ones(shape=dataBag[in1].value.shape) * dataBag[out].gradient)

    def test_backward_reduce_first_axis(self):
        in1 = Connection()
        out = Connection()
        operation = ReduceSumNode(in1, out, axis=0)
        dataBag = {in1: ConnectionData(value=np.array([[1., 2., 3.], [4., 5., 6.]])),
                   out: ConnectionData(gradient=2)}
        operation.forward_backward(dataBag)
        assert_array_equal(dataBag[in1].gradient, np.ones(shape=dataBag[in1].value.shape) * dataBag[out].gradient)

    def test_backward_reduce_second_axis(self):
        in1 = Connection()
        out = Connection()
        operation = ReduceSumNode(in1, out, axis=1)
        dataBag = {in1: ConnectionData(value=np.array([[1., 2., 3.], [4., 5., 6.]])),
                   out: ConnectionData(gradient=2)}
        operation.forward_backward(dataBag)
        assert_array_equal(dataBag[in1].gradient, np.ones(shape=dataBag[in1].value.shape) * dataBag[out].gradient)


class MatrixMultiplyNodesTests(unittest.TestCase):
    def test_forward(self):
        in1 = Connection()
        in2 = Connection()
        out = Connection()
        operation = MatrixMultiplyNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([[0., 1., 2.], [3., 4., 5.]])),
                   in2: ConnectionData(value=np.array([6., 7., 8.])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([23., 86.]))

    def test_backward(self):
        in1 = Connection()
        in2 = Connection()
        out = Connection()
        operation = MatrixMultiplyNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([[0., 1., 2.], [3., 4., 5.]])),
                   in2: ConnectionData(value=np.array([6., 7., 8.])[:, np.newaxis]),
                   out: ConnectionData(gradient=np.array([1., 1.])[:, np.newaxis])}
        operation.forward_backward(dataBag)
        grad = [
            [numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out], np.array([[1., 0., 0.], [0., 0., 0.]]))[0, 0],
             numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out], np.array([[0., 1., 0.], [0., 0., 0.]]))[0, 0],
             numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out], np.array([[0., 0., 1.], [0., 0., 0.]]))[0, 0]],
            [numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out], np.array([[0., 0., 0.], [1., 0., 0.]]))[1, 0],
             numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out], np.array([[0., 0., 0.], [0., 1., 0.]]))[1, 0],
             numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out], np.array([[0., 0., 0.], [0., 0., 1.]]))[1, 0]]
        ]
        assert_array_almost_equal(dataBag[in1].gradient, np.array(grad), 5)


class MaxNodeTests(unittest.TestCase):
    def test_forward(self):
        in1 = Connection()
        in2 = Connection()
        out = Connection()
        operation = MaxNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([-1., 1., 0.])),
                   in2: ConnectionData(value=np.array([1., 0., -5.])),
                   out: ConnectionData()}
        operation.forward(dataBag)
        assert_array_equal(dataBag[out].value, np.array([1., 1., 0.]))

    def test_backward(self):
        in1 = Connection()
        in2 = Connection()
        out = Connection()
        operation = MaxNode(in1, in2, out)
        dataBag = {in1: ConnectionData(value=np.array([-1., 1., 5.])),
                   in2: ConnectionData(value=np.array([1., 0., -5.])),
                   out: ConnectionData(gradient=np.array([7., 8., 9.]))}
        operation.forward_backward(dataBag)
        assert_array_almost_equal(dataBag[in1].gradient,
                                  numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in1], dataBag[out]))
        assert_array_almost_equal(dataBag[in2].gradient,
                                  numerical_gradient(lambda: operation.forward_backward(dataBag), dataBag[in2], dataBag[out]))


class BroadcastNodeTests(unittest.TestCase):
    def test_forward(self):
        in1 = Connection()
        out = Connection()
        operation = BroadcastNode(in1, out, axis=1)
        data_bag = {in1: ConnectionData(np.array([1., 2., 3.])),
                   out: ConnectionData()}
        operation.forward(data_bag)
        assert_array_equal(data_bag[out].value, np.array([[1.], [2.], [3.]]))

    def test_backward(self):
        in1 = Connection()
        out = Connection()
        operation = BroadcastNode(in1, out, axis=1)
        data_bag = {in1: ConnectionData(np.array([1., 2., 3.])),
                    out: ConnectionData(gradient=np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]))}
        operation.forward_backward(data_bag)
        assert_array_equal(data_bag[in1].gradient, np.array([3., 6., 9.]))
