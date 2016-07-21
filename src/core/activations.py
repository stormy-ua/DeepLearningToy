from simulation import *


def relu(ctx: SimulationContext, x: Connection, name=""):
    relu1 = ctx.max(x, ctx.constant(0, name))
    return relu1