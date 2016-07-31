from pydeeptoy.computational_graph import *


def relu(cg: ComputationalGraph, x: Connection, name=""):
    return cg.max(x, cg.constant(0))