import numpy as np
from nodes import *
from simulation import *


def sgd(network, X, one_hot_y, learning_rate=0.05, batch_size=250):
    indexes = np.arange(0, len(X))
    np.random.shuffle(indexes)
    train_x = X[indexes]
    train_y = one_hot_y[:, indexes]
    for batch in range(0, len(X), batch_size):
        batch_x = train_x[batch:batch + batch_size]
        batch_y = train_y[:, batch:batch + batch_size]
        network.forward_backward({ network.x_in: batch_x.T, network.one_hot_y_in: batch_y })
        for v in network.ctx.input_variables:
            v.value += -learning_rate * v.gradient
