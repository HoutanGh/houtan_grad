from graphviz import Digraph
from my_grad.value import Value

# function for traversing through computational graph created by Value objects

def trace(root):
    # building a set of all nodes and edges, similar to children method in class Value
    nodes, edges = set(), set()

    def build(x):
        if x not in nodes:
            nodes.add(x)
            for child in x._prev: # iterates through the list of the most recent values that make up x
                edges.add((child, x))
                build(child) 
        
    build(root)
    return nodes, edges

def draw_trace(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)

    for n in nodes:
        u_id = str(id(n)) # returns identity of object
        dot.node(name=u_id, label=f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}", shape='record')


        if n._op: # if an operation has been done, so its not the end of the graph
            dot.node(name = u_id + n._op, label= n._op)
            dot.edge(u_id + n._op, u_id)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# building a topological graph, through topological sorting

def build_topo(node):
    topo = []
    visited = set()
    
    def build(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)
    
    build(node)
    return topo

    