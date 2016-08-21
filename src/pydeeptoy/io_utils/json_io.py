from pydeeptoy.computational_graph import ComputationalGraph
import json
from itertools import chain


class JsonSerializer:
    @staticmethod
    def serialize(cg: ComputationalGraph):
        nodes = [{"name": str(n), "variable": "none"} for n in cg.nodes]

        links = list(chain.from_iterable([
                                             [{"source": cg.nodes.index(i), "target": cg.nodes.index(k)}
                                              for i in cg.adjacencyOutMap[c] for k in cg.adjacencyInMap[c]]
                                             for c in cg.adjacencyInMap if c in cg.adjacencyOutMap]))

        for c in cg.inputs:
            if c.name != "":
                n = {"name": c.name, "variable": "input"}
                nodes.append(n)
                links.append({"source": nodes.index(n), "target": cg.nodes.index(cg.adjacencyInMap[c][0])})

        for c in cg.outputs:
            n = {"name": c.name, "variable": "output"}
            nodes.append(n)
            links.append({"source": cg.nodes.index(cg.adjacencyOutMap[c][0]), "target": nodes.index(n)})

        j = json.dumps({"nodes": nodes, "links": links})

        return j
