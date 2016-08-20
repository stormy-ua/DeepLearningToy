from pydeeptoy.computational_graph import *


def softmax(cg: ComputationalGraph, x: Connection, name=""):
    exp1 = cg.exp(x)
    reduce_sum1 = cg.broadcast(cg.reduce_sum(exp1, axis=0), axis = 0)
    div1 = cg.div(exp1, reduce_sum1, name=name)
    return div1


def cross_entropy(cg: ComputationalGraph, x: Connection, one_hot_y: Connection, name=""):
    log1 = cg.multiply(cg.log(x), one_hot_y)
    mul2 = cg.multiply(log1, cg.constant(-1))
    reduce_sum3 = cg.reduce_sum(mul2)
    div2 = cg.div(reduce_sum3, cg.shape(x, 1))
    return div2


def hinge(cg: ComputationalGraph, x: Connection, one_hot_y: Connection, name=""):
    f2 = cg.multiply(x, one_hot_y)
    f3 = cg.reduce_sum(f2, axis=0)
    f4 = cg.sum(cg.constant(1), cg.sum(x, cg.multiply(cg.constant(-1), f3)))
    f5 = cg.multiply(f4, cg.sum(cg.constant(1), cg.multiply(cg.constant(-1), one_hot_y)))
    f6 = cg.max(f5, cg.constant(0))
    f7 = cg.reduce_sum(f6)
    f8 = cg.div(f7, cg.shape(x, 1))
    return f8
