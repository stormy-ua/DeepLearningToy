from simulation import *


def softmax(ctx: SimulationContext, x: Connection, one_hot_y: Connection, samples_count = 1):
    exp1 = ctx.exp(x)
    reduce_sum1 = ctx.reduce_sum(exp1, axis=0)
    mul1 = ctx.multiply(exp1, one_hot_y)
    reduce_sum2 = ctx.reduce_sum(mul1, axis=0)
    div1 = ctx.div(reduce_sum2, reduce_sum1)
    log1 = ctx.log(div1)
    mul2 = ctx.multiply(log1, Connection(-1))
    reduce_sum3 = ctx.reduce_sum(mul2)
    div2 = ctx.div(reduce_sum3, ctx.constant(samples_count))
    return div2
