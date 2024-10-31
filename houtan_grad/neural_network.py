import random
from my_grad.value import Value


# neural network library

# library class for defining zero_grad

class Library:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []

ACTIVATIONS = {
        'relu': lambda x: x.relu(),
        'tanh': lambda x: x.tanh()
        }
    
class Neuron(Library):
    def __init__(self, n_in, non_lin=''):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)] # wx_i + b for i inputs
        self.b = Value(random.uniform(-1, 1))
        self.non_lin = non_lin
    
    def __call__(self, x):
        act = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b) # sum takes second argument that is where it should start from
        # raw activation defined and now need to pass through non-linearity
        return ACTIVATIONS.get(self.non_lin, lambda x: x)(act)    
    # want to be able to carry out actions on all the parameters
    def parameters(self):
        return self.w +[self.b]
    
    def __repr__(self):
        return f"{self.non_lin.capitalize()}Neuron({len(self.w)})"
    

class Layer(Library):

    # adding **kwargs so that we can initialise neurons within Layers of the MLP with the activation functions defined as well as no activation function
    def __init__(self, n_in, n_out, **kwargs): # not just using non_lin so that we don't have a default activation function
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)] # creates the number of neurons to match n_out?

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Library): #multi-layer perceptron
    def __init__(self, n_in, n_outs, non_lin=''):
        size = [n_in] + n_outs # outs are already a list

        self.layers = [Layer(size[i], size[i+1], non_lin = non_lin) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in  layer.parameters()]
    
    def __repr__(self):
        return f"MLP: [{', '.join(str(layer) for layer in self.layers)}]"

class SGD:
    def __init__(self, parametersm lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0
