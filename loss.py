import numpy as np
from nodes import *
from simulation import *


def softmax(ctx: SimulationContext, x: Connection, one_hot_y: Connection):
    exp1 = ctx.exp(x)
    reduce_sum1 = ctx.reduce_sum(exp1, axis=0)
    mul1 = ctx.multiply(exp1, one_hot_y)
    reduce_sum2 = ctx.reduce_sum(mul1, axis=0)
    div1 = ctx.div(reduce_sum2, reduce_sum1)
    log1 = ctx.log(div1)
    mul2 = ctx.multiply(log1, Connection(-1))
    return mul2
