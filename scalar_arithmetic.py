from value import Value
from graphviz import Digraph
import matplotlib.pyplot as plt

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)  # __add__ method is called and new value created with 
           # (self, other) being the _children argument
d = a*b + c

# print(c)
# print(c._prev)
# print(d._prev) # but only gives most recent 

from helper_functions import trace, draw_trace

graph = draw_trace(d)
graph.render('graph', format='png', view=True)



