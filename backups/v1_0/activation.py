import numpy as np

def activation(z):

    return 1 / (1 + np.exp(-z))


def activation_deriv(z):
    
    s = 1 / (1 + np.exp(-z))

    return s * (1 - s)