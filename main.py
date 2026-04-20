import numpy as np
from network import Network as Network

data = np.load('mnist.npz')
x_train = data["x_train"] / 255 # NOTE normalizing the data here
y_train = data["y_train"]
x_test = data["x_test"] / 255
y_test = data["y_test"]


myNet = Network(shape=[784,128,64,10])
myNet.descend(x_train=x_train,y_train=y_train,alpha=0.5,max_steps=1000,batch_size=32)

