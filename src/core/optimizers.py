import numpy as np
from nodes import *
from simulation import *


def sgd(ctx: SimulationContext, x_in: Connection, one_hot_y_in: Connection, learningRate=0.05, batchSize=250):
    X = x_in.value.T
    one_hot_y = one_hot_y_in.value
    indexes = np.arange(0, len(X))
    np.random.shuffle(indexes)
    train_x = X[indexes]
    train_y = one_hot_y[:, indexes]
    for batch in range(0, len(X), batchSize):
        batch_x = train_x[batch:batch + batchSize]
        batch_y = train_y[:, batch:batch + batchSize]
        x_in.value = batch_x.T
        one_hot_y_in.value = batch_y
        ctx.forward_backward()
        for v in [n for n in ctx.inputs if isinstance(n, Variable)]:
            v.value += -learningRate * v.gradient
