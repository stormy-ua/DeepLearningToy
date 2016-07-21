from simulation import *


def relu(ctx: SimulationContext, x: Connection):
    relu1 = ctx.max(x, ctx.variable(0))
    return relu1