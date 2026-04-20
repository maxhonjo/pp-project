
'''
# ---------- RESOURCES ----------

https://cs231n.github.io/neural-networks-1/
    naming conventions

# ---------- INITIALIZATION ----------

Network(
    shape = [784, 128, 64, 10]
        list of integers
        specify the shape of the neural network, include the output and input layer

    random = True
        initialize weights and biases randomly

    TODO
    activation_hidden = 'relu'
    activation_output = 'sigmoid'


    TODO
    load = 'file.???'
        initialize weights and biases from a file

    TODO
    type = ???
        for non sequential integer outputs

    )

# ---------- FORWARD PROPAGATION ----------


Network.f_prop(
    x_input = np.array([0.99, 0.86, ..., 0.76, 0.50])
        numpy array with length
        NOTE: all training data must be normalized outside of the network in this implementation
        NOTE: possible feature is accepting any shape input and reshaping it to the vector
    )

    returns:
    a -> [np.arr(), np.arr(), np.arr()]
        python list containing numpy arrays with the activations of the neurons in the network
    z -> 



'''



import numpy as np
from activation_v2 import activate, af_deriv

class Network2:
    '''
    .__init__
    Network Initialization

    parameters:
        shape       | (list) | list containing the structre of the network, include input/output
        random_wb   | (bool) | randomize the weights and biases
        af_hidden   | (str)  | activation functions for the hidden layers
        af_output   | (str)  | activation function for the output layer

    '''
    def __init__(self, shape, random_wb = True, af_hidden='relu', af_output='softmax'):
        self.shape       = shape
        self.layers      = len(shape) - 1

        self.input_size  = shape[0]
        self.output_size = shape[-1]

        self.af_hidden = af_hidden
        self.af_output = af_output
        
        # TODO add other kinds of initialization other than random values
        if random_wb:
            self.init_random()

    '''
    .init_random
    Initialization when random_wb = True
    '''
    def init_random(self):
        self.weights = []
        self.biases  = []

        # iterate through every layer and randomize weights and biaes
        for i in range(self.layers):
            m = self.shape[i]     # number of neurons in previous layer
            n = self.shape[i+1]   # number of neurons in current layer

            scale         = np.sqrt(2.0 / m)                            # NOTE scale differs based on activation function?, better implementation?
            weight_matrix = np.random.uniform(-scale, scale, (n, m))    # 
            bias_vector   = np.zeros((n, 1))                            # 
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    '''
    .f_prop
    Forward Propagation Method

    parameters:
        x_input -> MUST BE NORMALIZED BEFORE PROPAGATION

    returns:
        z, a -> 
    
    NOTE:
    - input layer IS included with z, a (empty array for z)
    '''
    def f_prop(self, x_input):
        
        z = [np.array([])]                # input layer has no z, but adding a 0th element for index alignment with a
        a = [x_input.reshape(-1, 1)]      # a[0] is the input layer activation

        for layer_idx in range(self.layers):
            
            # use a_prev & weights -> z_next
            a_prev = a[layer_idx]
            z_next = self.weights[layer_idx].dot(a_prev) + self.biases[layer_idx]

            # activate z_next -> a_next
            if layer_idx != self.layers - 1: #
                a_next = activate(z_next, self.af_hidden)
            else:
                a_next = activate(z_next, self.af_output)

            z.append(z_next)
            a.append(a_next)

        return z, a
    
    '''
    .b_prop
    Backward Propagation Method
        z, a -> results from the forward propagation method
        y_true -> one-hot encoded expected vector NOTE must already be one-hot and (-1, 1) shape

        returns:


    '''
    def b_prop(self, z, a, y_true):

        dw = []
        db = []

        da = (a[-1] - y_true) * (2 / self.output_size) 
        for i in range(self.layers):
            
            layer_idx = -(i + 1)

            if i == 0:
                delta = da
            else:
                delta = af_deriv(z[layer_idx], self.af_hidden) * da

            a_prev = a[layer_idx - 1]
            dw.append(delta.dot(a_prev.T))
            db.append(delta)

            da = self.weights[layer_idx].T.dot(delta)

        dw.reverse()
        db.reverse()

        return dw, db




