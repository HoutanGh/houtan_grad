# houtan_grad
# Neural Network Library

This repository contains a simple neural network library implemented in Python, along with a custom `Value` class for automatic differentiation. It is purpose-built for scalar numbers and does not have implementation for tensors (I will try and do this later on in another library).

The library is in the subfolder my_grad and the other files outside this subfolder are implementations of the library and testing of its functionality.

## Files

- `neural_network.py`: Defines the neural network components.
- `value.py`: Implements the `Value` class for handling automatic differentiation.

## neural_network.py

### Library Class

Provides basic functionalities required for neural network operations.

- **Methods**:
  - `zero_grad()`: Resets the gradients of all parameters to zero.
  - `parameters()`: Returns an empty list. Meant to be overridden by subclasses.

### Neuron Class

Represents a single neuron in a neural network.

- **Constructor**:
  - `__init__(self, n_in, non_lin='')`: Initializes the neuron with random weights and an optional activation function (`relu` or `tanh`).

- **Methods**:
  - `__call__(self, x)`: Computes the neuron's output for input `x`.
  - `parameters()`: Returns the neuron's weights and bias.
  - `__repr__(self)`: Returns a string representation of the neuron.

### Layer Class

Represents a layer of neurons.

- **Constructor**:
  - `__init__(self, n_in, n_out, **kwargs)`: Initializes the layer with `n_out` neurons, each with `n_in` inputs.

- **Methods**:
  - `__call__(self, x)`: Computes the layer's output for input `x`.
  - `parameters()`: Returns all parameters of the neurons in the layer.
  - `__repr__(self)`: Returns a string representation of the layer.

### MLP Class

Represents a Multi-Layer Perceptron (MLP).

- **Constructor**:
  - `__init__(self, n_in, n_outs, non_lin='')`: Initializes the MLP with a list of layer sizes `n_outs` and an optional activation function.

- **Methods**:
  - `__call__(self, x)`: Computes the MLP's output for input `x`.
  - `parameters()`: Returns all parameters of the layers in the MLP.
  - `__repr__(self)`: Returns a string representation of the MLP.

## value.py

### Value Class

Represents a value in a computational graph, supporting automatic differentiation.

- **Constructor**:
  - `__init__(self, data, _children=(), _op='', label='')`: Initializes the value with data and optional children and operation label.

- **Methods**:
  - `__add__(self, other)`: Defines addition for `Value` objects.
  - `__mul__(self, other)`: Defines multiplication for `Value` objects.
  - `__pow__(self, other)`: Defines power operation for `Value` objects.
  - `__truediv__(self, other)`: Defines division for `Value` objects.
  - `tanh(self)`: Applies the tanh activation function.
  - `relu(self)`: Applies the ReLU activation function.
  - `exp(self)`: Applies the exponential function.
  - `backward(self)`: Computes the gradients through backpropagation.
  - `__repr__(self)`: Returns a string representation of the value.

## Usage

To use this library, import the necessary classes and build your neural network model. Here's a simple example:

```python
from neural_network import MLP
from value import Value

# Define a simple MLP with 2 inputs, one hidden layer with 3 neurons, and 1 output
mlp = MLP(2, [3, 1], non_lin='relu')

# Create input values
x = [Value(1.0), Value(2.0)]

# Compute the output of the MLP
output = mlp(x)

print(output)

