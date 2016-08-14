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
        one_hot_y = cg.constant(np.array([
            [0, 0, 1],
            [0, 0, 1]
        ]).T)

        loss = softmax(cg, x, one_hot_y, name="softmax")

        sc = SimulationContext()
        sc.forward(cg)

        self.assertAlmostEqual(sc[loss].value, 1.04, 3)
