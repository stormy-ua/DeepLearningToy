from pydeeptoy.simulation import *


class SgdOptimizer:
    def __init__(self, learning_rate=0.05):
        self._learning_rate = learning_rate

    def minimize(self, ctx: SimulationContext, cg: ComputationalGraph, params={}):
        ctx.forward_backward(cg, params)
        for v in cg.input_variables:
            ctx[v].value += -self._learning_rate * ctx[v].gradient


class MomentumSgdOptimizer:
    def __init__(self, learning_rate=0.05, momentum=0.9):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._vs = dict()

    def minimize(self, ctx: SimulationContext, cg: ComputationalGraph, params={}):
        ctx.forward_backward(cg, params)
        for v in cg.input_variables:
            if v not in self._vs:
                self._vs[v] = np.zeros(ctx[v].value.shape)
            self._vs[v] = self._momentum * self._vs[v] - self._learning_rate * ctx[v].gradient
            ctx[v].value += self._vs[v]
