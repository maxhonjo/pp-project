import numpy as np
from activation import activate, af_deriv
from utils import ONEHOT

class Network:

    '''Network Initialization

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
            self.randomize_wb()



    '''Random Initialization
    '''
    def randomize_wb(self):
        self.weights = []
        self.biases  = []

        # iterate through every layer and randomize weights and biaes
        for i in range(self.layers):
            m = self.shape[i]     # number of neurons in previous layer
            n = self.shape[i+1]   # number of neurons in current layer

            scale         = np.sqrt(2.0 / m)                            # NOTE scale differs based on activation function?, better implementation?
            weight_matrix = np.random.uniform(-scale, scale, (n, m))    # TODO initialize weights differently for each layer.
            bias_vector   = np.zeros((n, 1))                            # Initializing biases at zero is the standard. 
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)



    '''Forward Propagation Method

    parameters:
        x_input | (np.arr) | NORMALIZED input for the network

    returns:
        z, a    | (list)   | list of numpy arrays for z and a values. 
                           | NOTE that the input layer is included with z and a (empty for z)
    '''
    def f_prop(self, x_input, output_only=False):
        
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

        if output_only: # this is for testing the forward propagation
            return a[-1]

        return z, a
    


    '''Backward Propagation Method

    parameters:
        z, a   | (list)  | results from the forward propagation method
        y_true | (np.arr)| one-hot encoded expected vector NOTE must already be one-hot and (-1, 1) shape

    returns:
        dw, db | (list)  | list of numpy arrays containing the ascent gradient
    '''
    def b_prop(self, z, a, y_true):

        dw = [] #List of matrices for each layer
        db = [] #List of vectors for each layer

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



    '''Stochastic Descent
    
    parameters:
        x_train    | training data inputs that batch will randomly select from
        y_train    | training data labels ...
        alpha      | to calculate the step, alpha * gradient vectors
        max_steps  | number of batches
        batch_size | batch size
    '''
    def descend(self, x_train, y_train, alpha, max_steps=1000, batch_size=32):
        
        count = 0
        while count < max_steps:

            # get a batch of data from the training data
            batch_idxs = np.random.choice(len(x_train), batch_size, replace=False)
            x_batch = x_train[batch_idxs]
            y_batch = y_train[batch_idxs]

            # get the optimal descent for the batch
            dw_avg, db_avg, grad_mean = self.get_descent(x_train=x_batch, y_train=y_batch)
            
            # apply the descent
            for i in range(len(self.weights)):
                self.weights[i] -= dw_avg[i] * alpha
                self.biases[i]  -= db_avg[i] * alpha

            # finished descent, just visuals
            count += 1
            if count % 500 == 0:
                # print(f"step {count} | grad_mean: {grad_mean:.6f}")
                pass



    '''Get the Optimal Descent (given a batch)
    
    parameters:
        x_train | (np.arr) | numpy array with each input from the dataset
        y_train | (np.arr) | numpy array with the corresponding int

    returns:
        dw_avg    | (list)  | list containing numpy arrays with the change to the weights
        db_avg    | (list)  | list containing numpy arrays with the change to the biases
        grad_mean | (float) | represents the mean accross the above two
    
    NOTE the one hot encoding of the y_train is done at this step.
    '''
    def get_descent(self, x_train, y_train):

        batch_size = len(x_train)

        dw_sum = [np.zeros_like(w) for w in self.weights] #np.zeros_like returns an array of equal size of zeros.
        db_sum = [np.zeros_like(b) for b in self.biases]
        #print(gradient_vector)

        for i in range(batch_size):

            y_true = ONEHOT(y_train[i], self.output_size)
            z, a   = self.f_prop(x_input=x_train[i])
            dw, db = self.b_prop(z=z,
                                 a=a,
                                 y_true=y_true) #self.b_prop returns a list of matrices/vectors for the weights/biases of each layer.

            for j in range(self.layers):
                dw_sum[j] += dw[j]
                db_sum[j] += db[j]

        dw_avg = [dw / batch_size for dw in dw_sum]
        db_avg = [db / batch_size for db in db_sum]

        grad_mean = np.abs(np.concatenate([d.flatten() for d in dw_avg + db_avg])).mean()

        return dw_avg, db_avg, grad_mean


