import matplotlib.pyplot as plt
import numpy as np
import math
from   tensorflow.examples.tutorials.mnist import input_data

def get_ordered(X_train):
    ordered = [ 
            X_train[7] , # 0 
            X_train[4] , # 1 
            X_train[16], # 2
            X_train[1] , # 3 
            X_train[2] , # 4
            X_train[27], # 5
            X_train[3] , # 6
            X_train[14], # 7 
            X_train[5] , # 8 
            X_train[8] , # 9
            ]       
    return ordered

def print_images(ordered,m,n):
    f, ax = plt.subplots(m,n)
    ordered = get_ordered(X_train);
    for i in range(m):
        for j in range(n):
            ordered[i*n+j] = ordered[i*n+j].reshape(28,28)
            ax[i][j].imshow(ordered[i*n+j], cmap = plt.cm.binary, interpolation="nearest")
            ax[i][j].axis("off")

    plt.show()

def softmax_func(x):
    x_exp = [math.exp(i) for i in x]
    sum_x_exp = sum(x_exp)
    softmax = [i / sum_x_exp for i in x_exp]
    return softmax 

def softmax_derivative(x):
    x_exp = [math.exp(i) for i in x]
    sum_x_exp = sum(x_exp)
    softmax = [i / sum_x_exp for i in x_exp]
    return softmax 

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return (x*(1-x))

class activation_function:
    def __init__(self,function,derivative):
        self.function = function
        self.derivative = derivative

    def function(self,x):
        return self.function(x) 

    def derivative(self,x):
        return self.derivative(x) 

class neural_network:
    def __init__(self, num_outputs, layers, output_layer, momentum):
        self.num_outputs = num_outputs
        self.num_layers= len(layers)
        self.layers = layers
        self.momentum = momentum
        self.output_layer = output_layer
        self.__set_layer_sigma()

    def __set_layer_sigma(self):
        # set up initial guesses for first layer
        self.layers[0].sigma = np.sqrt(float(2) /
            (self.layers[0].num_inputs + self.layers[1].num_inputs))

        # set initial guesses for hidden layers - 1
        for i in range(1,len(self.layers)-1):
            self.layers[i].sigma = np.sqrt(float(2) /
            (self.layers[i].num_inputs + self.layers[i+1].num_inputs))

        # set initial guesses for last hidden layer
        self.layers[len(self.layers)-1].sigma = np.sqrt(float(2) /
        (self.layers[len(self.layers)-1].num_inputs + self.num_outputs))

    def backward_prop(self,x,yhat,y):
        eta = 0.5
        der_out = yhat-y

        # get derivatives for all layers
        for layer in self.layers:
            layer.find_neuron_derivative()
         
        # find the derivative with respect to another variable
        # for j in range(len(self.layers)):
        # print(self.layers[1].neurons[0].current_weight)

        for i in range(self.layers[1].num_neurons):
            for j in range(len(self.layers[1].neurons[0].current_weight)):
                if j is 0: 
                    der_neur = der_out[i]*self.layers[1].neurons[i].derivative*1
                else:
                    der_neur = der_out[i]*self.layers[1].neurons[i].derivative*self.layers[0].neurons[j-1].output
        
                self.layers[1].neurons[i].new_weight[j] = self.layers[1].neurons[i].current_weight[j] - eta * der_neur

        print(self.layers[1].neurons[0].new_weight[0])
        print(self.layers[1].neurons[0].new_weight[1])
        print(self.layers[1].neurons[0].new_weight[2])
        
        print(self.layers[1].neurons[1].new_weight[0])
        print(self.layers[1].neurons[1].new_weight[1])
        print(self.layers[1].neurons[1].new_weight[2])

        # print(self.layers[1].neurons[0].new_weight)
        print("backward prop")

    def train_network(self, x, y):
        yhat = np.array(self.forward_prop(x))
        y = np.array(y).reshape((len(y),1))
        error = 0.5*(yhat - y)*(yhat - y)
        # print(error)
        self.backward_prop(x,yhat,y)

    def __layer_opperations(self,X,layers):
        for neuron in layers.neurons:
            neuron.input = np.dot(X,neuron.current_weight)
            neuron.output = self.layers[0].activation.function(neuron.input)

        output = [layers.neurons[i].output for i in range(layers.num_neurons)]
        return output

    def forward_prop(self, data):
        X = [1] + data
        
        layer_input = self.__layer_opperations(X,self.layers[0])
        for i in range(1,len(self.layers)):
            layer_input = [1] + layer_input
            layer_input = np.array(self.__layer_opperations(layer_input,self.layers[i]) )
        return layer_input

    def repot_error(self, yhat, y):
        print("verify accuracy")

class layer:
    def __init__(self, num_neurons, num_inputs, activation):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.sigma = None
        self.activation = activation

        self.neurons = [neuron(self.num_inputs,self.sigma) 
                for i in range(self.num_neurons)]

    def find_neuron_derivative(self):
        for neuron in self.neurons:
            neuron.derivative = self.activation.derivative(neuron.output)

class neuron:
    def __init__(self,num_inputs,sigma):
        self.output = None
        self.input = None
        self.current_weight = np.random.normal(0,sigma,[num_inputs+1,1])
        self.new_weight = [None]*(num_inputs+1)
        self.derivative = None
             

num_inputs = 2 
num_outputs = 2
momentum = None

# create the sigmoid activation function
sigmoid = activation_function(sigmoid_func,sigmoid_derivative)
softmax = activation_function(softmax_func,softmax_func)

# create layer structure
layer0 = layer(2,num_inputs,sigmoid)
layer1 = layer(2,layer0.num_neurons,sigmoid)
hidden_layers = [layer0, layer1]

# create neural network framework
network = neural_network(num_outputs,hidden_layers,softmax,momentum)

# create testing initial parameters
network.layers[0].neurons[0].current_weight[0] = 0.35
network.layers[0].neurons[0].current_weight[1] = 0.15
network.layers[0].neurons[0].current_weight[2] = 0.20

network.layers[0].neurons[1].current_weight[0] = 0.35
network.layers[0].neurons[1].current_weight[1] = 0.25
network.layers[0].neurons[1].current_weight[2] = 0.30

network.layers[1].neurons[0].current_weight[0] = 0.60
network.layers[1].neurons[0].current_weight[1] = 0.40
network.layers[1].neurons[0].current_weight[2] = 0.45

network.layers[1].neurons[1].current_weight[0] = 0.60
network.layers[1].neurons[1].current_weight[1] = 0.50
network.layers[1].neurons[1].current_weight[2] = 0.55

x = [0.05, 0.1]
y = [0.01, 0.99]
network.train_network(x,y)

# print(network.layers[0].neurons[0].input)
# print(network.layers[0].neurons[0].output)
# print(network.layers[0].neurons[1].input)
# print(network.layers[0].neurons[1].output)
# print("\n")
#
# print(network.layers[1].neurons[0].input)
# print(network.layers[1].neurons[0].output)
# print(network.layers[1].neurons[1].input)
# print(network.layers[1].neurons[1].output)
# print("\n")

# print(network.layers[0].neurons[0].derivative)
