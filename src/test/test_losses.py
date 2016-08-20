from pydeeptoy.simulation import *
import unittest
from numpy.testing import *
from pydeeptoy.losses import *


class SoftmaxTests(unittest.TestCase):
    def test_forward(self):
        cg = ComputationalGraph()
        x = cg.constant(np.array([
            [-2.85, 0.86, 0.28],
            [-2.85, 0.86, 0.28]
        ]).T, name="x")
        sm = softmax(cg, x, name="softmax")

        sc = SimulationContext()
        sc.forward(cg)

        assert_array_almost_equal(sc[sm].value, [[0.016, 0.016], [0.631, 0.631], [0.353, 0.353]], 3)


class CrossEntropyTests(unittest.TestCase):
    def test_forward(self):
        cg = ComputationalGraph()
        x = cg.constant(np.array([
            [0.016, 0.016], [0.631, 0.631], [0.353, 0.353]
        ]), name="x")
        one_hot_y = cg.constant(np.array([
            [0, 0, 1],
            [0, 0, 1]
        ]).T)

        loss = cross_entropy(cg, x, one_hot_y, name="cross_entropy")

        sc = SimulationContext()
        sc.forward(cg)

        assert_array_almost_equal(sc[loss].value, 1.04, 3)

    def test_forward_with_softmax(self):
        cg = ComputationalGraph()
        x = cg.constant(np.array([
            [-2.85, 0.86, 0.28],
            [-2.85, 0.86, 0.28]
        ]).T, name="x")
        one_hot_y = cg.constant(np.array([
            [0, 0, 1],
            [0, 0, 1]
        ]).T)

        loss = cross_entropy(cg, softmax(cg, x, name="softmax"), one_hot_y, name="cross_entropy")

        sc = SimulationContext()
        sc.forward(cg)

        self.assertAlmostEqual(sc[loss].value, 1.04, 3)
