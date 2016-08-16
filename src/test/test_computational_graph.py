from pydeeptoy.simulation import *
import unittest
from numpy.testing import *


class ConvolutionTests(unittest.TestCase):
    def test_conv2d(self):
        cg = ComputationalGraph()
        x = cg.constant(np.array([[
            [
                [1, 2, 0, 2, 2],
                [1, 2, 1, 2, 2],
                [1, 0, 2, 0, 0],
                [1, 0, 1, 1, 0],
                [1, 2, 0, 1, 1]
            ],
            [
                [1, 0, 2, 2, 2],
                [1, 1, 1, 2, 2],
                [0, 1, 1, 2, 2],
                [2, 0, 0, 0, 2],
                [0, 2, 0, 2, 0]
            ],
            [
                [2, 0, 2, 1, 1],
                [0, 1, 0, 2, 0],
                [0, 0, 0, 2, 1],
                [2, 0, 1, 1, 1],
                [0, 1, 0, 0, 1]
            ]
        ]]), name="x")
        w = cg.constant(np.array([
            [1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, -1, 0, 0, 0, -1, 1, 1, -1, 1, 0, 0, 0],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 0, -1, -1, 0, 0, 1, 1, 0, 1, -1, -1, 0]
        ]).transpose(1, 0), name="w")

        conv = cg.conv2d(x, w, receptive_field_size=3, filters_number=2, stride=2, padding=1)

        sc = SimulationContext()
        sc.forward(cg)

        conv_value = sc.data_bag[conv].value
        assert_array_equal(conv_value, np.array(
            [[
                [
                    [0, 0, 0],
                    [2, 10, 5],
                    [0, 2, 3]
                ],
                [
                    [-1, 0, 1],
                    [1, 14, 3],
                    [3, 3, 5]
                ]
            ]]
        ))
        self.assertEqual(conv_value.shape, (sc[x].value.shape[0], sc[w].value.shape[1], 3, 3))

    def test_tmp(self):
        cg = ComputationalGraph()
        x_in = cg.constant(np.ones(shape=(1, 1, 28, 28)))
        w_in = 0.01*np.random.randn(16, 3)
        conv = cg.conv2d(x_in, cg.constant(w_in), receptive_field_size=4, filters_number=3, stride=2, padding=0)
        ctx = SimulationContext()
        #ctx.forward(cg)

        assert 0==0