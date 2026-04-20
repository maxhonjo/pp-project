import numpy as np

def activation(z):
    return np.maximum(0, z)          # ReLU

def activation_deriv(z):
    return (z > 0).astype(float)     # 1 where z > 0, else 0

def softmax(z):
    e = np.exp(z - np.max(z))        # subtract max for numerical stability
    return e / e.sum()


class Network:

    '''
    Network Initialization Parameters:
        layerCount (int) -> number of layers in the network
        neuronShape [(int), (int), (int)] -> array contiaining the number of neurons for each layer
        trainX [[],[],[]] -> numpy array containing each k observation. 
        trainY [int, int, int] -> numpy array containing integer values of the 'correct' output.


    '''
    def __init__(self, hiddenLayerCount, hiddenShape, x_train, y_train):

        # Assining attributes from object init
        self.hiddenLayerCount = hiddenLayerCount
        self.hiddenShape = hiddenShape
        self.x_train = x_train / np.max(x_train)      # NOTE Normalizing x_train
        self.y_train = y_train

        # Separate initialization
        self.init_networkShapes()   # Initialize inputLayerSize, outputLayerSize, neuronShape
        self.init_wb()              # Initialization of the Network based on the above configurations




    '''
    initialize attributes:
        self.inputLayerSize
        self.outputLayerSize
        self.neuronShape
    '''
    def init_networkShapes(self):
        # Initializing the Neuron Shape
        self.inputLayerSize = self.x_train[0].flatten().size
        self.outputLayerSize = np.unique(self.y_train).size

        self.neuronShape = [self.inputLayerSize]
        self.neuronShape.extend(self.hiddenShape)
        self.neuronShape.append(self.outputLayerSize)

    '''
    initialize attributes:
        self.weights
        self.biases
    '''
    def init_wb(self):
        self.weights = []
        self.biases = []
        for i in range(self.hiddenLayerCount + 1):
            m = self.neuronShape[i]
            n = self.neuronShape[i + 1]

            scale = np.sqrt(2.0 / m)   # He init — the only change from Xavier
            weightMatrix = np.random.uniform(-scale, scale, (n, m))
            biasVector = np.zeros((n, 1))

            self.weights.append(weightMatrix)
            self.biases.append(biasVector)

    '''
    backpropagation function parameters:
        alpha (float) -> scales the descent gradient before updating the network
        threshold (float) -> if the means of the descent gradients is smaller
    '''
    def descend(self, alpha, threshold = 0.000001, max_steps=50, batch_size=32):
        dw_mean, db_mean = self.descend_one(alpha=alpha, batch_size=batch_size)
        count = 1
        while (dw_mean > threshold) or (db_mean > threshold):
            dw_mean, db_mean = self.descend_one(alpha=alpha, batch_size=batch_size)
            count += 1
            # print(f'step {count}  dw={dw_mean:.6f}  db={db_mean:.6f}')
            if count == max_steps:
                print(f'quit at {max_steps} steps.')
                break
        print('done descending.')
        

    def descend_one(self, alpha, batch_size=32):

        indices = np.random.choice(len(self.x_train), batch_size, replace=False)
        gradients = []
        for k in indices:
            z, a = self.forward_propagate(self.x_train[k])
            dw, db = self.backward_propagate(x=self.x_train[k],
                                            y=self.y_train[k],
                                            a=a,
                                            z=z)
            gradients.append((dw, db))

        dw_avg = [
            np.mean(np.stack([g[0][i] for g in gradients], axis=0), axis=0)
            for i in range(self.hiddenLayerCount + 1)
        ]
        db_avg = [
            np.mean(np.stack([g[1][i] for g in gradients], axis=0), axis=0)
            for i in range(self.hiddenLayerCount + 1)
        ]

        for i in range(len(self.weights)):
            self.weights[i] -= dw_avg[i] * alpha
            self.biases[i]  -= db_avg[i] * alpha

        dw_mean = np.abs(np.concatenate([dw.flatten() for dw in dw_avg])).mean()
        db_mean = np.abs(np.concatenate([db.flatten() for db in db_avg])).mean()
        return dw_mean, db_mean
            


    def forward_propagate(self, k):

        z_all = []
        a_all = []

        a = k.flatten().reshape(self.inputLayerSize, 1)

        for currentLayer in range(self.hiddenLayerCount + 1):

            z = self.weights[currentLayer].dot(a) + self.biases[currentLayer]
            if currentLayer == self.hiddenLayerCount:  # output layer
                a = softmax(z)
            else:                                       # hidden layers
                a = activation(z)                       # ReLU

            z_all.append(z)
            a_all.append(a)


        return z_all, a_all

    def backward_propagate(self, x, y, a, z):

        dw_all = []
        db_all = []

        # ─── One-hot encode the expected output ───────────────────────────────
        true_vector = np.zeros((self.outputLayerSize, 1))
        true_vector[y] = 1

        # ─── Output layer delta ───────────────────────────────────────────────
        # dL/da for MSE loss
        da = (a[-1] - true_vector) * (2 / self.outputLayerSize)

        # ─── Backprop through each layer (output → input) ─────────────────────
        for i in range(self.hiddenLayerCount + 1):  # i=0 is output layer

            layer_idx = -(i + 1)   # -1, -2, -3 ...  indexes into weights/z/a from the back

            delta = activation_deriv(z[layer_idx]) * da        # shape: (n, 1)

            # Activation from the layer BELOW (i.e. the inputs to this weight matrix)
            if i == self.hiddenLayerCount:
                a_below = x.flatten().reshape(-1, 1)          # raw input, for weights[0]
            else:
                a_below = a[layer_idx - 1]                    # previous hidden activation

            dw = delta.dot(a_below.T)                         # shape: (n, m)
            db = delta                                        # shape: (n, 1)

            dw_all.append(dw)
            db_all.append(db)

            # Propagate da to the layer below through this weight matrix
            da = self.weights[layer_idx].T.dot(delta)         # shape: (m, 1)

        # Reverse so index 0 = first layer (matching self.weights order)
        dw_all.reverse()
        db_all.reverse()

        return dw_all, db_all