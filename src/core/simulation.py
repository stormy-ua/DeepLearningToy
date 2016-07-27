from computational_graph import *


class SimulationContext:
    @staticmethod
    def get_adjacent_nodes(cg: ComputationalGraph, node: Node):
        return [cg.adjacencyOutMap[output] for output in node.outputs if output in cg.adjacencyOutMap]

    def sort_topologically(self, cg: ComputationalGraph):
        sorted_nodes = []

        def depth_first_search(on_vertex_finished):
            discovered = dict()
            finished = dict()

            def visit(vertex, time):
                time += 1
                discovered[vertex] = time

                for v in self.get_adjacent_nodes(cg, vertex):
                    if v not in discovered:
                        time = visit(v, time)

                time += 1
                finished[vertex] = time
                on_vertex_finished(time, vertex)
                return time

            time = 0
            for v in cg.nodes:
                if v not in discovered:
                    time = visit(v, time)

        depth_first_search(lambda time, node: sorted_nodes.append(node))

        return sorted_nodes

    def forward(self, cg: ComputationalGraph, params=dict()):
        for p, v in params.items():
            p.value = v

        [node.forward() for node in self.sort_topologically(cg)]

    def backward(self, cg: ComputationalGraph):
        for i in cg.outputs:
            i.reset_gradient(to_value=1)
        [node.backward() for node in reversed(self.sort_topologically(cg))]

    def forward_backward(self, cg: ComputationalGraph, params=dict()):
        self.forward(cg, params)
        self.backward(cg)


