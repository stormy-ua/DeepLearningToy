from pydeeptoy.computational_graph import *
from itertools import takewhile
from itertools import chain


class SimulationContext:
    def __init__(self):
        self.data_bag = dict()

    def get_data(self, key):
        if key not in self.data_bag:
            self.data_bag[key] = ConnectionData(value=key.init_value)

        return self.data_bag[key]

    def __getitem__(self, key):
        return self.get_data(key)

    def __setitem__(self, key, value):
        self.data_bag[key] = value

    @staticmethod
    def sort_topologically(cg: ComputationalGraph, out=list()):
        sorted_nodes = []

        def depth_first_search(on_vertex_finished):
            discovered = dict()
            finished = dict()

            def visit(vertex, time):
                time += 1
                discovered[vertex] = time

                for v in cg.get_adjacent_in_nodes(vertex):
                    if v not in discovered:
                        time = visit(v, time)

                time += 1
                finished[vertex] = time
                on_vertex_finished(time, vertex)
                return time

            time = 0
            root_nodes = chain.from_iterable([cg.adjacencyOutMap[c] for c in out]) if len(out) > 0 else cg.nodes
            for v in root_nodes:
                if v not in discovered:
                    time = visit(v, time)

        depth_first_search(lambda time, node: sorted_nodes.insert(0, node))

        sorted_nodes.reverse()
        return sorted_nodes

    def forward(self, cg: ComputationalGraph, params=dict(), out=list()):
        for p, v in params.items():
            self.get_data(p).value = v

        for node in self.sort_topologically(cg, out):
            node.forward(self)

    def backward(self, cg: ComputationalGraph, reset_gradient=True, out=list()):
        if reset_gradient:
            for i in cg.outputs:
                self.get_data(i).reset_gradient(to_value=1)
        [node.backward(self) for node in reversed(self.sort_topologically(cg))]

    def forward_backward(self, cg: ComputationalGraph, params=dict(), reset_gradient=True, out=list()):
        self.forward(cg, params, out=out)
        self.backward(cg, reset_gradient=reset_gradient, out=out)


