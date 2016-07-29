import numpy as np
from nodes import *
from simulation import *


class SgdOptimizer:
    def __init__(self, learning_rate=0.05):
        self._learning_rate = learning_rate

    def minimize(self, ctx: SimulationContext, cg: ComputationalGraph, params={}):
        ctx.forward_backward(cg, params)
        for v in cg.input_variables:
            ctx[v].value += -self._learning_rate * ctx[v].gradient
