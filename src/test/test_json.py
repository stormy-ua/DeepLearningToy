from pydeeptoy.simulation import *
import unittest
from numpy.testing import *
from pydeeptoy.losses import *
from pydeeptoy.io_utils.json_io import JsonSerializer
from sklearn import datasets
from pydeeptoy.asserts import *
from pydeeptoy.networks import *
from pydeeptoy.optimizers import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
import unittest
from pydeeptoy.losses import *


class JsonSerializerTests(unittest.TestCase):
    def test_serialization(self):
        cg = ComputationalGraph()
        x_in = cg.constant(name="X.T")
        nn_output = softmax(cg, neural_network(cg, x_in, 10, 10, 10, 10, 3), name="nn_output")

        y_train = cg.constant(name="one_hot_y")
        loss = cross_entropy(cg, nn_output, y_train, "loss_cross_entropy")

        #g = cg.div(cg.sum(cg.constant(1), cg.constant(2)), cg.constant(3))

        serializer = JsonSerializer()
        json = serializer.serialize(cg)
