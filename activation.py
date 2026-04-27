import numpy as np

'''
Activation & Derivative Functions
activate(z, 'function type')
af_deriv(z, 'function type')

Parameters:
    z | (np arr) | numpy array with pre-activatino values
    f | (str)    | function type (lowercase string) ex. 'relu'

Returns:
    f(z) | (np arr) | numpy array of activated values.
'''
def activate(z, f = None): 

    all_functions = ['relu', 'sigmoid', 'softmax']

    if f not in all_functions:
        raise Exception(f'activation function not in {all_functions}')

    if f == 'relu':
        return relu(z)

    if f == 'sigmoid':
        return sigmoid(z)
    
    if f == 'softmax':
        return softmax(z)
    


def af_deriv(z, f):
    if f == 'relu':
        return relu_deriv(z)
    
    if f == 'sigmoid':
        raise Exception('sigmoid derivative not implemented')
    

'''
ACTIVATION FUNCTIONS
'''
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    raise Exception('sigmoid not implemented')    

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

'''
ACTIVATION FUNCTION DERIVATIVES
'''

def relu_deriv(z):
    return (z > 0).astype(float)


    