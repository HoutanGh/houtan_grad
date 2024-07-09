from graphviz import Digraph
from value import Value

def trace(root):
    # building a set of all nodes and edges, similar to children method in class Value
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev: # interates through the list