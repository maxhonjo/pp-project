import numpy as np


def ONEHOT(y, output_size):
    y_true = np.zeros((output_size, 1))
    y_true[y] = 1

    return y_true