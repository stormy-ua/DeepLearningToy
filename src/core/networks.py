from simulation import *
import numpy as np
from activations import *
from losses import *

def neural_network_2layer(ctx: SimulationContext, n1, n2, X, one_hot_y):
    # constants
    x_in = ctx.constant(X.T)
    one_hot_y_in = ctx.constant(one_hot_y)
    # variables
    W1 = 0.01 * np.random.randn(n1, X.shape[1])
    W2 = 0.01 * np.random.randn(n2, n1)
    b1 = 0.01 * np.ones(n1)
    b2 = 0.01 * np.ones(n2)
    w1_in = ctx.variable(W1)
    w2_in = ctx.variable(W2)
    b1_in = ctx.variable(b1[:, np.newaxis])
    b2_in = ctx.variable(b2[:, np.newaxis])
    # 1st layer
    matrix_mul1 = ctx.matrix_multiply(w1_in, x_in)
    sum1 = ctx.sum(matrix_mul1, b1_in)
    relu1 = relu(ctx, sum1)
    # 2nd layer
    matrix_mul2 = ctx.matrix_multiply(w2_in, relu1)
    sum2 = ctx.sum(matrix_mul2, b2_in)
    relu2 = relu(ctx, sum2)
    # loss
    loss = softmax(ctx, relu2, one_hot_y_in, X.shape[1])
    return (w1_in, w2_in, b1_in, b2_in, loss)