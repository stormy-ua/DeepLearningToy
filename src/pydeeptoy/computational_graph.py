from pydeeptoy.nodes import *
from itertools import chain


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

    def get_adjacent_out_nodes(self, node: Node):
        return chain.from_iterable(
            [self.adjacencyInMap[output] for output in node.outputs if output in self.adjacencyInMap])

    def get_adjacent_in_nodes(self, node: Node):
        return chain.from_iterable(
            [self.adjacencyOutMap[input] for input in node.inputs if input in self.adjacencyOutMap])

    def add_input_connection(self, connection: Connection, operation: Node):
        if connection not in self.adjacencyInMap:
            self.adjacencyInMap[connection] = list()
        self.adjacencyInMap[connection].append(operation)

    def add_output_connection(self, connection: Connection, operation: Node):
        if connection not in self.adjacencyOutMap:
            self.adjacencyOutMap[connection] = list()
        self.adjacencyOutMap[connection].append(operation)

    @staticmethod
    def variable(name="", init_value=None, shape=None):
        return Variable(name=name, init_value=init_value, shape=shape)

    @staticmethod
    def constant(value=None, name=""):
        return Constant(value, name=name)

    def add_unary_op(self, op, in1: Connection):
        out = Connection()
        operation = op(in1, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_output_connection(out, operation)
        return out

    def add_binary_op(self, op, in1: Connection, in2: Connection, name=""):
        out = Connection(name=name)
        operation = op(in1, in2, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_input_connection(in2, operation)
        self.add_output_connection(out, operation)
        return out

    def sum(self, in1: Connection, in2: Connection):
        return self.add_binary_op(SumNode, in1, in2)

    def multiply(self, in1: Connection, in2: Connection):
        return self.add_binary_op(MultiplyNode, in1, in2)

    def matrix_multiply(self, in1: Connection, in2: Connection):
        return self.add_binary_op(MatrixMultiplyNode, in1, in2)

    def div(self, in1: Connection, in2: Connection, name=""):
        return self.add_binary_op(DivNode, in1, in2, name=name)

    def exp(self, in1: Connection):
        return self.add_unary_op(ExpNode, in1)

    def sqrt(self, in1: Connection):
        return self.add_unary_op(SqrtNode, in1)

    def log(self, in1: Connection):
        return self.add_unary_op(LogNode, in1)

    def eval(self, in1: Connection, expression):
        out = Connection()
        operation = ExpressionNode(in1, out, expression)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_output_connection(out, operation)
        return out

    def shape(self, in1: Connection, axis=0):
        return self.eval(in1, lambda x: x.shape[axis])

    def reduce_sum(self, in1: Connection, axis=None):
        out = Connection()
        operation = ReduceSumNode(in1, out, axis)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_output_connection(out, operation)
        return out

    def max(self, in1: Connection, in2: Connection):
        return self.add_binary_op(MaxNode, in1, in2)

    def broadcast(self, in1: Connection, axis=1):
        out = Connection()
        operation = BroadcastNode(in1, out, axis)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_output_connection(out, operation)
        return out

    def transpose(self, in1: Connection, *axes):
        out = Connection()
        operation = TransposeNode(in1, out, axes)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_output_connection(out, operation)
        return out

    def reshape(self, in1: Connection, newaxes, name=""):
        out = Connection(name=name)
        operation = ReshapeNode(in1, out, newaxes)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_output_connection(out, operation)
        return out

    def tensor_3d_to_cols(self, in1: Connection, receptive_field_size, stride=1, padding=1, name=""):
        out = Connection(name=name)
        operation = Tensor3dToCol(in1, out, receptive_field_size, stride=stride, padding=padding)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_output_connection(out, operation)
        return out

    def conv2d(self, x_in: Connection, w_in: Connection, receptive_field_size, filters_number, stride=1, padding=1,
               name=""):
        """
        Computes a 2-D convolution given 4-D input and filter tensors.
        """
        x_cols = self.tensor_3d_to_cols(x_in, receptive_field_size, stride=stride, padding=padding)
        mul = self.transpose(self.matrix_multiply(x_cols, w_in), 0, 2, 1)

        #output_width = self.sum(self.div(self.sum(self.sum(self.shape(x_in, 2), self.constant(-1 * receptive_field_size)),
        #                        self.constant(2 * padding)), self.constant(stride)), self.constant(1))
        # output_height = (h - f + 2 * p) / s + 1

        output = self.reshape(mul, (-1, filters_number, receptive_field_size, receptive_field_size))

        output.name = name
        return output
