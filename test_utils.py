import numpy as np
import matplotlib.pyplot as plt
from network import Network


# Default Dataset for Testing Functions
data = np.load('datasets/mnist.npz')
# data = np.load('datasets/fashion_mnist.npz')

x_train = data["x_train"] / 255 # NOTE normalizing the data here
y_train = data["y_train"]
x_test = data["x_test"] / 255
y_test = data["y_test"]


'''Test Accuracy
parameters:
    network | (Network class) | network to test the accuracy of
    x_test  | (np array)      | standard format set of x values for testing
    y_test  | (np array)      | standard format set of y values for testing

returns:
    accuracy | between 0-1 proportion of test set that was correct
'''
def test_accuracy(network, x_test=x_test, y_test=y_test, test_size=500):

    test_idxs = np.random.choice(len(x_test), test_size,replace=False)
    x_test = x_test[test_idxs]
    y_test = y_test[test_idxs]

    results = np.zeros_like(x_test)

    for i in range(len(x_test)):
        output = network.f_prop(x_test[i], output_only=True)
        
        prediction = np.argmax(output)
        true_val = y_test[i]
    
        if prediction == true_val:
            results[i] = 1

    accuracy = np.mean(results)
    # print(f'network accuracy: {accuracy * 100:.2f}%')
    return np.mean(results)


'''Best of X Networks
parameters:
    n | (int) | number of networks to create and test
    remaining parameters are the same as .descend_stoc(), they will be used to train the networks before comparing

returns:
    Network object with the best accuracy of the group of n (after completing .descend_stoc() described above).
'''
def best_of(n=20, shape=[784, 128, 64, 10], xt=x_train, yt=y_train, alpha=0.5, max_steps=500, batch_size=32):

    networks = []
    scores = np.zeros(n)
    for i in range(n):
        # Create one new Network
        net = Network(shape=shape)

        # Train the network 1000x
        net.descend_stoc(x_train=x_train, y_train=y_train, alpha=alpha, max_steps=max_steps, batch_size=32)
        
        # Add network score to list
        score = test_accuracy(net)
        scores[i] = score

        print(f'network {i} score: {score}')

        # Add network to list
        networks.append(net)

    best_idx = np.argmax(scores)
    return networks[best_idx]


'''Show One
Copied from the MNIST lab in programming class.
'''
def show_one(image, label=None, pred=None):
    """Display one 28x28 image."""
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")

    if label is not None and pred is not None:
        title = f"true = {label} | pred = {pred}"
        plt.title(title)
    elif label is not None:
        plt.title(f"true = {label}")
    elif pred is not None:
        plt.title(f"pred = {pred}")

    plt.axis("off")
    plt.show()


'''Fashion Mapping
FASHION_MAP[y_train_idx] == 'name of clothing element'
'''
FASHION_MAP = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


'''Counter Generator
'''
def make_counter():
    n = 0
    while True:
        yield n
        n += 1
COUNTER = make_counter()