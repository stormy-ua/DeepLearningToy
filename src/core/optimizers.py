import numpy as np
from nodes import *
from simulation import *


def sgd(ctx, network, X, one_hot_y, learning_rate=0.05, batch_size=250):
    indexes = np.arange(0, len(X))
    np.random.shuffle(indexes)
    train_x = X[indexes]
    train_y = one_hot_y[:, indexes]
    for batch in range(0, len(X), batch_size):
        batch_x = train_x[batch:batch + batch_size]
        batch_y = train_y[:, batch:batch + batch_size]
        ctx.forward_backward(network, {network.x_in: batch_x.T, network.one_hot_y_in: batch_y})
        for v in network.ctx.input_variables:
            v.value += -learning_rate * v.gradient


class SgdOptimizer:
    def __init__(self, model: ComputationalGraph, cost: ComputationalGraph):
        assert len(cost.outputs) == 1
        self._cost = cost
        self._model = model

    def minimize(self, ctx: SimulationContext,
                 input_connection: Connection, target_connection: Connection,
                 train_input, train_target, learning_rate=0.05, batch_size=250):
        indexes = np.arange(0, len(train_input))
        np.random.shuffle(indexes)
        train_x = train_input[indexes]
        train_y = train_target[:, indexes]
        for batch in range(0, len(train_x), batch_size):
            batch_x = train_x[batch:batch + batch_size]
            batch_y = train_y[:, batch:batch + batch_size]
            ctx.forward(self._model, {input_connection: batch_x.T})
            ctx.forward_backward(self._cost, {target_connection: batch_y})
            ctx.backward(self._model, reset_gradient=False)
            for v in self._model.input_variables:
                ctx[v].value += -learning_rate * ctx[v].gradient