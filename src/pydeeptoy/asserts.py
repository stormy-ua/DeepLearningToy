from pydeeptoy.nodes import *


def numerical_gradient(forward_backward, input: ConnectionData, output: ConnectionData, mask = 1):
    dx = 1E-8
    forward_backward()
    x = output.value
    input.value += dx * mask
    forward_backward()
    xdx = output.value
    grad = (xdx - x) / dx * output.gradient
    return grad
