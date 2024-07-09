from graphviz import Digraph

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if not isinstance(other, Value):
            return NotImplemented
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            return NotImplemented
        out = Value(self.data * other.data, (self, other), '*')
        return out

def trace(root):
    nodes, edges = set(), set()

    def build(x):
        if x not in nodes:
            nodes.add(x)
            for child in x._prev:
                edges.add((child, x))
                build(child)
    
    build(root)
    return nodes, edges

def draw_trace(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = trace(root)

    for n in nodes:
        u_id = str(id(n))
        label = f"data {n.data:.4f}"
        dot.node(name=u_id, label=label, shape='record')
        
        if n._op:
            op_id = u_id + '_op'
            dot.node(name=op_id, label=n._op)
            dot.edge(op_id, u_id)

    for n1, n2 in edges:
        u_id1 = str(id(n1))
        u_id2 = str(id(n2))
        op_id2 = u_id2 + '_op' if n2._op else u_id2
        dot.edge(u_id1, op_id2)

    return dot

# Example usage
a = Value(2)
b = Value(3)
c = a + b * a

dot = draw_trace(c)
dot.render('graph', format='svg', view=True)
