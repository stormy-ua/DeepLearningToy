from nodes import *


class ComputationalGraph:
    def __init__(self):
        self.nodes = []
        self.adjacencyInMap = dict()
        self.adjacencyOutMap = dict()

    @property
    def outputs(self):
        return [i for i in self.adjacencyOutMap if i not in self.adjacencyInMap]

    @property
    def inputs(self):
        return [i for i in self.adjacencyInMap if i not in self.adjacencyOutMap]

    @property
    def input_variables(self):
        return [n for n in self.inputs if isinstance(n, Variable)]

    def get_adjacent_nodes(self, node: Node):
        return [self.adjacencyOutMap[output] for output in node.outputs if output in self.adjacencyOutMap]

    def add_input_connection(self, connection: Connection, operation: Node):
        self.adjacencyInMap[connection] = operation

    @staticmethod
    def variable(value=None, name=""):
        return Variable(value, name=name)

    @staticmethod
    def constant(value=None, name=""):
        return Constant(value, name=name)

    def add_unary_op(self, op, in1: Connection):
        out = Connection()
        operation = op(in1, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.adjacencyOutMap[out] = operation
        return out

    def add_binary_op(self, op, in1: Connection, in2: Connection):
        out = Connection()
        operation = op(in1, in2, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_input_connection(in2, operation)
        self.adjacencyOutMap[out] = operation
        return out

    def sum(self, in1: Connection, in2: Connection):
        return self.add_binary_op(SumNode, in1, in2)

    def multiply(self, in1: Connection, in2: Connection):
        return self.add_binary_op(MultiplyNode, in1, in2)

    def matrix_multiply(self, in1: Connection, in2: Connection):
        return self.add_binary_op(MatrixMultiplyNode, in1, in2)

    def div(self, in1: Connection, in2: Connection):
        return self.add_binary_op(DivNode, in1, in2)

    def exp(self, in1: Connection):
        return self.add_unary_op(ExpNode, in1)

    def log(self, in1: Connection):
        return self.add_unary_op(LogNode, in1)

    def reduce_sum(self, in1: Connection, axis = None):
        out = Connection()
        operation = ReduceSumNode(in1, out, axis)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.adjacencyOutMap[out] = operation
        return out

    def max(self, in1: Connection, in2: Connection):
        return self.add_binary_op(MaxNode, in1, in2)

    def broadcast(self, in1: Connection, axis=1):
        out = Connection()
        operation = BroadcastNode(in1, out, axis)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.adjacencyOutMap[out] = operation
        return out


