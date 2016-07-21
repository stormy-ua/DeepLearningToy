import unittest
from nodes import *
import numpy as np
from numpy.testing import *


def numericalGradient(operation, input: Connection, output: Connection, mask = 1):
    dx = 1E-8
    operation.forward()
    operation.backward()
    x = output.value
    input.value += dx * mask
    operation.forward()
    operation.backward()
    xdx = output.value
    return (xdx - x) / dx * output.gradient
