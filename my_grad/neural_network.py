import random
from value import Value


class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)] # wx_i + b for i inputs
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        act = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b) # sum takes second argument that is where it should start from
        # raw activation defined and now need to pass through non-linearity
        out = act.tanh()
        return out
    
class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]


a = Neuron(2)
b = Layer(2, 3)
print(b.neurons)
    
        
