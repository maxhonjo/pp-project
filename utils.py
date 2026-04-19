import numpy as np

def relu_derivative(z):

    relu_z = np.where(z > 0, 1, 0)
    return relu_z



def print_rounded(matrix):

    out = np.round(matrix, 2)
    print(out)

