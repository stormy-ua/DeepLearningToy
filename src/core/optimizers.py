import numpy as np
from nodes import *
from simulation import *


def sgd(network, X, one_hot_y, learningRate=0.05, batchSize=250):
    indexes = np.arange(0, len(X))
    np.random.shuffle(indexes)
    train_x = X[indexes]
    train_y = one_hot_y[:, indexes]
    for batch in range(0, len(X), batchSize):
        batch_x = train_x[batch:batch + batchSize]
        batch_y = train_y[:, batch:batch + batchSize]
        network.forward_backward(batch_x, batch_y)
        for v in network.ctx.input_variables:
            v.value += -learningRate * v.gradient
