import numpy as np

def activation(z):
    return np.maximum(0, z)          # ReLU

def activation_deriv(z):
    return (z > 0).astype(float)     # 1 where z > 0, else 0

def softmax(z):
    e = np.exp(z - np.max(z))        # subtract max for numerical stability
    return e / e.sum()
