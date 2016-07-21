import numpy as np
from nodes import *


class SimulationContext:
    nodes = []
    adjacencyInMap = dict()
    adjacencyOutMap = dict()

    def get_adjacent_nodes(self, node: Node):
        return [self.adjacencyOutMap[output] for output in node.outputs if output in self.adjacencyOutMap]

    def sort_topologically(self):
        sorted_nodes = []

        def depth_first_search(on_vertex_finished):
            discovered = dict()
            finished = dict()

            def visit(vertex, time):
                time += 1
                discovered[vertex] = time

                for v in self.get_adjacent_nodes(vertex):
                    if v not in discovered:
                        time = visit(v, time)

                time += 1
                finished[vertex] = time
                on_vertex_finished(time, vertex)
                return time

            time = 0
            for v in self.nodes:
                if v not in discovered:
                    time = visit(v, time)

        depth_first_search(lambda time, node: sorted_nodes.append(node))

        return sorted_nodes

    def forward(self):
        [node.forward() for node in self.sort_topologically()]

    def backward(self):
        for i in [i for i in self.adjacencyOutMap if i not in self.adjacencyInMap]:
            i.reset_gradient(1)
        [node.backward() for node in reversed(self.sort_topologically())]

    def add_input_connection(self, connection: Connection, operation: Node):
        self.adjacencyInMap[connection] = operation

    def sum(self, in1: Connection, in2: Connection):
        out = Connection()
        operation = SumNode(in1, in2, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_input_connection(in2, operation)
        self.adjacencyOutMap[out] = operation
        return out

    def multiply(self, in1: Connection, in2: Connection):
        out = Connection()
        operation = MultiplyNode(in1, in2, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_input_connection(in2, operation)
        self.adjacencyOutMap[out] = operation
        return out

    def div(self, in1: Connection, in2: Connection):
        out = Connection()
        operation = DivNode(in1, in2, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.add_input_connection(in2, operation)
        self.adjacencyOutMap[out] = operation
        return out

    def exp(self, in1: Connection):
        out = Connection()
        operation = ExpNode(in1, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.adjacencyOutMap[out] = operation
        return out

    def log(self, in1: Connection):
        out = Connection()
        operation = LogNode(in1, out)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.adjacencyOutMap[out] = operation
        return out

    def reduce_sum(self, in1: Connection, axis = None):
        out = Connection()
        operation = ReduceSumNode(in1, out, axis)
        self.nodes.append(operation)
        self.add_input_connection(in1, operation)
        self.adjacencyOutMap[out] = operation
        return out
