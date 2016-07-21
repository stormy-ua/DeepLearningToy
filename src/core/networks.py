from simulation import *
import numpy as np
from activations import *
from losses import *

def neural_network_2layer(ctx: SimulationContext, n1, n2, x_in, one_hot_y_in):
    # variables
    W1 = 0.01 * np.random.randn(n1, x_in.value.shape[0])
    W2 = 0.01 * np.random.randn(n2, n1)
    b1 = 0.01 * np.ones(n1)
    b2 = 0.01 * np.ones(n2)
    w1_in = ctx.variable(W1, "W1")
    w2_in = ctx.variable(W2, "W2")
    b1_in = ctx.variable(b1, "b1")
    b2_in = ctx.variable(b2, "b2")
    # 1st layer
    matrix_mul1 = ctx.matrix_multiply(w1_in, x_in)
    b1_in_broadcasted = ctx.broadcast(b1_in, axis=1)
    sum1 = ctx.sum(matrix_mul1, b1_in_broadcasted)
    relu1 = relu(ctx, sum1)
    # 2nd layer
    matrix_mul2 = ctx.matrix_multiply(w2_in, relu1)
    b2_in_broadcasted = ctx.broadcast(b2_in, axis=1)
    sum2 = ctx.sum(matrix_mul2, b2_in_broadcasted)
    relu2 = relu(ctx, sum2, "output")
    # loss
    loss = softmax(ctx, relu2, one_hot_y_in, x_in.value.shape[0])
    loss.name = "loss"
    return ([w1_in, w2_in], [b1_in, b2_in], relu2, loss)