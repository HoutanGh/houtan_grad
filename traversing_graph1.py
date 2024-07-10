from my_grad.value import Value
from graphviz import Digraph
import matplotlib.pyplot as plt

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')  # __add__ method is called and new value created with 
           # (self, other) being the _children argument
d = a*b; d.label = 'd'
e = d + c; e.label = 'e'
f = Value(-2.0,  label='f')

L = e * f; L.label = 'L'
# print(c)
# print(c._prev)
# print(d._prev) # but only gives most recent 

from helper_functions import trace, draw_trace

graph = draw_trace(L)
graph.render('graph_L', format='png', view=True)

from helper_functions import build_topo


topo = build_topo(L)
print(topo)
L.grad = 1.0
print(L._prev)
L._backward()



for node in reversed(topo):
    node._backward()

graph = draw_trace(L)
graph.render('graph_with_grads', format='png', view=True)

