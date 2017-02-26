import numpy as np


def sigmoid(z):
    """Return the output of the sigmoid function applied on z."""
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    """Return the output of the derivative of the sigmoid function applied on z."""
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    """Return the output of the hyperbolic tangent function applied on z."""
    return np.tanh(z)

def d_tanh(z):
    """Return the output of the derivative of the hyperbolic tangent function applied on z."""
    return 1 - tanh(z)**2
