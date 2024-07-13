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
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        # need to make a case for non-value objects
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward # assigning this nested function to the self._backward from __init__
        
        return out
    
    def __pow__(self, other):
        assert isinstance(self, (int, float)) # accomodating for non-float can be difficult
        out = Value(self.data**other, (self, ), f"**{other}")

        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other): # if can't do 2 * a, python will check if it can do a * 2
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad # using t**2 is probably faster
        out._backward = _backward

        return out
    
    # another popular activation fucntion, defined a the positive part of its argument

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad # returns 0 if in the negative part of the relu function
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
    
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


    
    

    
    
    