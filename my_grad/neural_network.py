import random
from value import Value

# neural network library

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
        self.neurons = [Neuron(n_in) for _ in range(n_out)] # creates the number of neurons to match n_out?

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MPL: #multi-layer perceptron
    def __init__(self, n_in, n_outs):
        size = [n_in] + n_outs # outs are already a list

        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
