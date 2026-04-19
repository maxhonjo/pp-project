import numpy as np
from network import Network

import matplotlib.pyplot as plt




# LOADING DATA 
data = np.load('mnist.npz')
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

# print("x_train:", x_train.shape, x_train.dtype)
# print("y_train:", y_train.shape, y_train.dtype)
# print("x_test: ", x_test.shape, x_test.dtype)
# print("y_test: ", y_test.shape, y_test.dtype)


# myNetwork = Network(hiddenLayerCount=2,
#                     hiddenShape=[256,128],
#                     x_train=x_train,
#                     y_train=y_train
#                     )


'''
TESTING FUNCTION: use to get the proportion correct of the test data of that network
'''
def test_network(network, x_test, y_test):

    results = np.zeros_like(x_test)

    for i in range(len(x_test)):
        z, a = network.forward_propagate(x_test[i] / 255)
        prediction = np.argmax(a[-1])
        true_val = y_test[i] 
    
        if prediction == true_val:
            results[i] = 1

    return np.mean(results)

'''
TESTING: one network
'''
myNetwork = Network(hiddenLayerCount=2, hiddenShape=[128, 64], x_train=x_train, y_train=y_train)
myNetwork.descend(alpha=0.5, threshold=0.0000000001, max_steps=10000)

print(test_network(myNetwork, x_test=x_test, y_test=y_test))




'''
TESTING: creating 10 networks, and then training the best one

'''
# TEST_NETWORK_COUNT = 30
# STEP_SIZE = 1000

# testNetworks = {}
# for i in range(TEST_NETWORK_COUNT):
#     testNetworks[i] = Network(hiddenLayerCount=2, hiddenShape=[128, 64], x_train=x_train, y_train=y_train)
#     testNetworks[i].descend(alpha=10.0, threshold=0.00000000001, max_steps=STEP_SIZE, batch_size=32)


# resultDict = {}
# for i in range(TEST_NETWORK_COUNT):
#     resultDict[i] = test_network(testNetworks[i], x_test=x_test, y_test=y_test)

# bestNetwork = testNetworks[np.argmax(np.array(resultDict.values()))]

# bestNetwork.descend(alpha=10.0, threshold=0.00000000001, max_steps=10000, batch_size=32)


# prop_correct = test_network(bestNetwork, x_test=x_test, y_test=y_test)
# print('\nPOST-DESCENT')
# print(f'{prop_correct * 100}% correct')



'''
TESTING: visualizing the diminishing returns of descent steps

'''
# myNetwork = Network(hiddenLayerCount=2, hiddenShape=[128, 64], x_train=x_train, y_train=y_train)

# count = 0
# prop_correct_list = []
# for i in range(50):

#     myNetwork.descend(alpha=10.0, threshold=0.00000000001, max_steps=100, batch_size=32)
#     prop_correct = test_network(myNetwork, x_test=x_test, y_test=y_test)
#     print(f'prop correct after {count} x 100 descents: {prop_correct}')

#     prop_correct_list.append(prop_correct)
#     count += 1


# x = [q for q in range(len(prop_correct_list))]

# plt.scatter(x=x, y=prop_correct_list)
# plt.show()