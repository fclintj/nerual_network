# Clint Ferrin
# Oct 12, 2017
# Neural Network Classifier

import matplotlib.pyplot as plt
import numpy as np
import pickle
from   tensorflow.examples.tutorials.mnist import input_data
import time

def main():
    num_inputs = 784 
    num_outputs= 10 
    batch_size = 100
    epochs = 10
    mse_freq = 50

    # open mnist data
    X,Y,X_test,Y_test = get_mnist_train("./data")

    # initialize activation functions
    relu = activation_function(relu_func,relu_der)
    sig  = activation_function(sigmoid_func,sigmoid_der)
    no_activation = activation_function(return_value,return_value)
    
    num_neurons = 300
    # two hidden layers
    layers1 = [layer(num_inputs,num_neurons,relu)]
    layers1.append(layer(num_neurons,100,sig))
    layers1.append(layer(100,num_outputs,no_activation))

    # create neural network
    network = NeuralNetwork(layers,eta=0.9,momentum=0.8,softmax=True) 

    # train network
    network.train_network(X,Y,batch_size=batch_size,
                          epochs=epochs,MSE_freq=mse_freq,reg=0.01)
