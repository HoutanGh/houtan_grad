import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        # Want to keep track of all the data of values that make current data, so introduce _children

        self.data = data
        self._prev = set(_children) # This is for efficiency, removing duplicates
        self._op = _op
        self.label = label
        self._backward = lambda: None
        self.grad = 0
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad # using t**2 is probably faster
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)
        build(self)

        self.grad = 1.0
        for x in reversed(topo):
            x._backward()


    
    

    
    
    