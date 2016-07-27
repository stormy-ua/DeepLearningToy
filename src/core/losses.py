from computational_graph import *


def softmax(cg: ComputationalGraph, x: Connection, one_hot_y: Connection, samples_count = 1, name=""):
    exp1 = cg.exp(x)
    reduce_sum1 = cg.reduce_sum(exp1, axis=0)
    mul1 = cg.multiply(exp1, one_hot_y)
    reduce_sum2 = cg.reduce_sum(mul1, axis=0)
    div1 = cg.div(reduce_sum2, reduce_sum1)
    log1 = cg.log(div1)
    mul2 = cg.multiply(log1, cg.constant(-1))
    reduce_sum3 = cg.reduce_sum(mul2)
    div2 = cg.div(reduce_sum3, cg.constant(samples_count))
    return div2
