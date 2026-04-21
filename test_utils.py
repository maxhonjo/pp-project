import numpy as np

data = np.load('mnist.npz')
x_train = data["x_train"] / 255 # NOTE normalizing the data here
y_train = data["y_train"]
x_test = data["x_test"] / 255
y_test = data["y_test"]

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