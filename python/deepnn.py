# Clint Ferrin
# Oct 12, 2017
# Neural Network Classifier

import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import time
from   tensorflow.examples.tutorials.mnist import input_data

def main():
    test_bilinear()    
    # test_number()
    # softmax_test()
    
def softmax_test():
    num_inputs = 3
    num_outputs = 3
    batch_size = 1
    epochs = 1 
    momentum = 0 

    # create the sigmoid activation function
    sigmoid = activation_function(sigmoid_func,sigmoid_derivative)
    softmax = activation_function(stable_softmax_func,softmax_respect_der)
    relu = activation_function(relu_func,relu_der)
    no_activation = activation_function(return_value,return_value)

    # create layer structure
    layer0 = layer(num_inputs,3,relu)
    layer1 = layer(3,3,sigmoid)
    layer3 = layer(3,3,no_activation)

    hidden_layers = [layer0, layer1, layer3]

    # create neural network framework
    # network = pickle.load(open("./classasgntrain1.p","rb"))


    network = neural_network(num_outputs,hidden_layers,"softmax",momentum)
    network.set_initial_conditions_3()
    
    # data = np.loadtxt("./data/classasgntrain1.dat",dtype=float)
    # x0 = data[:,0:2]
    # x1 = data[:,2:4]
    # data = data_frame(x0,x1)

    x = [0.1,0.2,0.7]
    y = [1,0,0]
    network.train_data(x,y)
    print(network.layers[0].neurons[0].output)
    # network.print_weights()
    # network.plot_error_array()

    # yhat = network.classify_data(data.xtot)
    # print(yhat)
    # y = np.r_[np.ones([data.N0,1]),np.zeros([data.N1,1])] 
    # num_err = sum(abs(yhat - y))
    # print("Percent of errors: %.4f"%(float(num_err)/data.N))
    #
    # test_data = data_frame(gendata2(0,10000),gendata2(1,10000))
    # yhat = network.classify_data(test_data.xtot)
    # num_err = sum(abs(yhat - test_data.y))
    # print("Percent of errors: %.5f"%(float(num_err)/test_data.N))
    #
    # plot_boundaries(data,network.forward_prop) 


def test_bilinear():
    num_inputs = 2 
    num_outputs = 2
    batch_size = 100
    epochs = 10 
    momentum = 0 

    # create the sigmoid activation function
    sigmoid = activation_function(sigmoid_func,sigmoid_derivative)
    softmax = activation_function(stable_softmax_func,softmax_respect_der)
    relu = activation_function(relu_func,relu_der)
    no_activation = activation_function(return_value,return_value)

    # create layer structure
    layer0 = layer(num_inputs,2,relu)
    layer1 = layer(2,2,sigmoid)
    hidden_layers = [layer0, layer1]

    # create neural network framework
    # network = pickle.load(open("./classasgntrain1.p","rb"))


    network = neural_network(num_outputs,hidden_layers,"none",momentum)
    network.set_initial_conditions()
    # x = np.array([0.05,0.10])
    # y = np.array([0, 1])

    data = np.loadtxt("./data/classasgntrain1.dat",dtype=float)
    x0 = data[:,0:2]
    x1 = data[:,2:4]
    data = data_frame(x0,x1)

    print(data.xtot.shape)
    print(data.class_tot)
    network.train_network(data.xtot,data.class_tot,batch_size,epochs)

    network.plot_error_array()

    yhat = network.classify_data(data.xtot)
    y = np.r_[np.ones([data.N0,1]),np.zeros([data.N1,1])] 
    num_err = sum(abs(yhat - y))
    print("Percent of errors: %.4f"%(float(num_err)/data.N))

    test_data = data_frame(gendata2(0,10000),gendata2(1,10000))
    yhat = network.classify_data(test_data.xtot)
    num_err = sum(abs(yhat - test_data.y))
    print("Percent of errors: %.5f"%(float(num_err)/test_data.N))

    plot_boundaries(data,network.forward_prop) 

def test_number():
    num_inputs = 2 
    num_outputs = 2
    batch_size = 5000
    epochs = 5 
    momentum = 0

    # create the sigmoid activation function
    sigmoid = activation_function(sigmoid_func,sigmoid_derivative)
    softmax = activation_function(stable_softmax_func,softmax_respect_der)
    no_activation = activation_function(return_value,return_value)

    # create layer structure
    layer0 = layer(num_inputs,2,sigmoid)
    layer1 = layer(2,2,no_activation)
    hidden_layers = [layer0, layer1]

    # create neural network framework
    # network = pickle.load(open("./classasgntrain1.p","rb"))

    network = neural_network(num_outputs,hidden_layers,"softmax",momentum)
    network.set_initial_conditions()
    # x = np.array([0.05,0.10])
    # y = np.array([0, 1])

    x = np.array(
        [[0.10, 0.05],
         [0.03, 0.1],
         [0.07, 0.3],
         [0.02, 0.4]])

    y = np.array(
        [[0, 1],
         [1, 0],
         [1, 0],
         [1, 0]])

    network.train_network(x,y,batch_size,epochs)
    print(network.classify_data(x))

def plot_data(data):
    fig = plt.figure() # make handle to save plot 
    plt.scatter(data.x0[:,0],data.x0[:,1],c='red',label='$x_0$')
    plt.scatter(data.x1[:,0],data.x1[:,1],c='blue',label='$x_1$')
    plt.xlabel('X Coordinate') 
    plt.ylabel('Y Coordinate') 
    plt.legend()

def plot_boundaries(data,equation):
    xp1 = np.linspace(data.xlim[0],data.xlim[1], num=100)
    yp1 = np.linspace(data.ylim[0],data.ylim[1], num=100) 
    
    red_pts = np.array([[],[]])
    blue_pts= np.array([[],[]])
    for x in xp1:
        for y in yp1:
            prob = equation([x,y])
            if prob[0] > prob[1]: 
                blue_pts = np.c_[blue_pts,[x,y]]
            else:
                red_pts = np.c_[red_pts,[x,y]]

    plot_data(data)
    plt.scatter(blue_pts[0,:],blue_pts[1,:],color='blue',s=0.25)
    plt.scatter(red_pts[0,:],red_pts[1,:],color='red',s=0.25)
    plt.xlim(data.xlim)
    plt.ylim(data.ylim)
    plt.show()

def gendata2(class_type,N):
    m0 = np.array(
         [[-0.132,0.320,1.672,2.230,1.217,-0.819,3.629,0.8210,1.808, 0.1700],
          [-0.711,-1.726,0.139,1.151,-0.373,-1.573,-0.243,-0.5220,-0.511,0.5330]])

    m1 = np.array(
          [[-1.169,0.813,-0.859,-0.608,-0.832,2.015,0.173,1.432,0.743,1.0328],
          [ 2.065,2.441,0.247,1.806,1.286,0.928,1.923,0.1299,1.847,-0.052]])

    x = np.array([[],[]])
    for i in range(N):
        idx = np.random.randint(10)
        if class_type == 0:
            m = m0[:,idx]
        elif class_type == 1:
            m = m1[:,idx]
        else:
            print("not a proper classifier")
            return 0 
        x = np.c_[x, [[m[0]],[m[1]]] + np.random.randn(2,1)/np.sqrt(5)]
    return x.T

class data_frame:
    def __init__(self, data0, data1):
        self.x0 = data0 
        self.x1 = data1 
        self.xtot = np.r_[self.x0,self.x1]
        self.N0 = self.x0.shape[0]
        self.N1 = self.x1.shape[0]
        self.N = self.N0 + self.N1
        self.xlim = [np.min(self.xtot[:,0]),np.max(self.xtot[:,0])]
        self.ylim = [np.min(self.xtot[:,1]),np.max(self.xtot[:,1])]
        class_x0 = np.c_[np.zeros([self.N0,1]),np.ones([self.N0,1])] 
        class_x1 = np.c_[np.ones([self.N1,1]),np.zeros([self.N1,1])] 
        self.class_tot = np.r_[class_x0,class_x1]
        self.y = np.r_[np.ones([self.N0,1]),np.zeros([self.N1,1])] 

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

def return_value(x):
    return x


def relu_func(x):
    return np.maximum(0,x)
    
def relu_der(x):
    if x > 0:
        return x
    else:
        return 0

def stable_softmax_func(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def softmax_func(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def softmax_respect_der(y_out,x,x_element):
    # for i, y in enumerate(y_out):
    #     if i is x_element:
    #         return y*(1-x)
    #     else:
    return -x*y

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
        self.total_error = None
        self.__set_layer_sigma()
        self.error_array = [] 

    def __set_layer_sigma(self):
        for i in range(len(self.layers)-1):
        # set initial guesses for hidden layers - 1
            self.layers[i].sigma = np.sqrt(float(2) /
            (self.layers[i].num_inputs + self.layers[i+1].num_inputs))
            for j in range(self.layers[i].num_neurons):
                self.layers[i].neurons[j].current_weight = np.random.normal(0,self.layers[i].sigma,[self.layers[i].num_inputs+1,1])

        # set initial guesses for last hidden layer
        for j in range(self.layers[len(self.layers)-1].num_neurons):
            self.layers[len(self.layers)-1].neurons[j].current_weight = np.random.normal(0,self.layers[i].sigma,[self.layers[len(self.layers)-1].num_inputs+1,1])

    def __update_weights(self):
        for i in range(self.num_layers):
            for j in range(self.layers[i].num_neurons):
                for k in range(len(self.layers[i].neurons[j].current_weight)):
                    self.layers[i].neurons[j].current_weight[k] = self.layers[i].neurons[j].new_weight[k]
                    
    def __layer_opperations(self,X,layer):
        for i in range(layer.num_neurons):
            for j in range(X.shape[0]):
                layer.neurons[i].weight_der[j] = X[j]
            layer.neurons[i].net = np.dot(X.T,layer.neurons[i].current_weight)

        net = np.empty([layer.num_neurons,1]) 
        for i in range(layer.num_neurons):
            net[i] = layer.neurons[i].net
            layer.neurons[i].output = layer.activation.function(net[i])

        output = layer.activation.function(net)
        return output 

    def train_network(self, x, y, batch_size, epochs):
        count = 0
        batch = np.random.randint(0,len(x),batch_size)
        for i in range(epochs):
            for sample in batch:
                self.train_data(x[sample],y[sample]) 
                if i%10 is 0:
                    self.error_array.append(self.total_error)

            print("Total error: %f"%(self.total_error))

    def plot_error_array(self):
        plt.plot(self.error_array) 
        plt.show()

    def train_data(self, x, y):
        yhat = np.array(self.forward_prop(x))
        yhat = np.array(yhat).reshape((len(yhat),1))
        y = np.array(y).reshape((len(y),1))
        self.backward_prop(x,yhat,y)
        self.total_error = sum(0.5*(yhat - y)*(yhat - y))

    def forward_prop(self, data):
        X = np.array(data).reshape((len(data),1))
        X = np.r_[[[1]],X]
        layer_output = self.__layer_opperations(X,self.layers[0])
        for i in range(1,len(self.layers)):
            layer_output = np.r_[[[1]],layer_output]
            layer_output = self.__layer_opperations(layer_output,self.layers[i])

        if self.output_layer == "softmax":
            layer_output = stable_softmax_func(layer_output)

        return np.array(layer_output).reshape(len(layer_output))
    
    def backward_prop(self,x,yhat,y):
        eta = 1 
        momentum_calculation = 0
        der_out = yhat-y
        # get derivatives for all layers
        for layer in self.layers:
            layer.find_neuron_derivatives()

        der_err_tot = 0
        # find weights for last layer if softmax
        if self.output_layer == "softmax":
            neuron_sum = 0
            for neuron in self.layers[self.num_layers-1].neurons:
                neuron_sum += np.exp(neuron.net) 

            for i, neuron in enumerate(self.layers[self.num_layers-1].neurons):
                for j in range(len(neuron.weight_der)):
                    der_neur = 0
                    for k in range(self.num_outputs):
                        if i is k:
                            der_neur += der_out[k] * (np.exp(neuron.net)*(neuron_sum-np.exp(neuron.net)))/(neuron_sum*neuron_sum)
                            # der_neur += der_out[k] * ((1-1.0/np.exp(neuron_sum))) * neuron.weight_der[j]
                        else:
                            der_neur += der_out[k] * np.exp(neuron.net) 
                            # der_neur += der_out[k] * (-1.0/np.exp(neuron_sum)) * neuron.weight_der[j]
                    # update new weight
                    momentum_calculation = self.momentum * momentum_calculation + eta * der_neur
                    neuron.new_weight[j] = neuron.current_weight[j] - momentum_calculation 

             # find derivative of total output error with respect to penultimate layer
            # der_err_tot += der_out[0] * (1-1.0/np.exp(neuron_sum)) * self.layers[self.num_layers-1].neurons[0].weight_der[1]

            der_out[0] * (np.exp(self.layers[self.num_layers-1].neurons[0].net) * (neuron_sum-np.exp(self.layers[self.num_layers-1].neurons[0].net)))/(neuron_sum*neuron_sum)

            for i in range(1,len(der_out)):
                der_err_tot += der_out[i] * (np.exp(self.layers[self.num_layers-1].neurons[i].net) *(neuron_sum-np.exp(self.layers[self.num_layers-1].neurons[i].net)))/(neuron_sum*neuron_sum)

        # find weights for last layer
        else: 
            for j in range(self.layers[self.num_layers-1].num_neurons):
                for k in range(len(self.layers[self.num_layers-1].neurons[0].current_weight)):
                    der_neur = self.layers[self.num_layers-1].neurons[0].weight_der[k] * self.layers[self.num_layers-1].neurons[j].derivative*der_out[j]

                    # update new weight
                    momentum_calculation = self.momentum * momentum_calculation + eta * der_neur
                    self.layers[self.num_layers-1].neurons[j].new_weight[k] = self.layers[self.num_layers-1].neurons[j].current_weight[k] - momentum_calculation 

            # find derivative of total output error with respect to penultimate layer
            for i in range(len(der_out)):
                der_err_tot += self.layers[self.num_layers-1].neurons[i].current_weight[1] * self.layers[self.num_layers-1].neurons[i].derivative*der_out[i]
        
        # find derivative of total output error with respect to weights in hidden layers 
        for i in range(self.num_layers-2,-1,-1):
            for j in range(self.layers[i].num_neurons):
                for k in range(len(self.layers[i].neurons[0].current_weight)):
                    der_neur = self.layers[i].neurons[j].derivative * self.layers[i].neurons[j].weight_der[k] * der_err_tot

                    # update new weight
                    momentum_calculation = np.array(self.momentum * momentum_calculation) + np.array(eta) * der_neur
                    self.layers[i].neurons[j].new_weight[k] = self.layers[i].neurons[j].current_weight[k] - momentum_calculation

            # calculate new total multiplicative derivative for next layer
            der_err_tot *= self.layers[i].neurons[0].weight_der[1]

        self.__update_weights()

    def classify_data(self, x):
        output = np.empty([x.shape[0],1]) 
        for i in range(x.shape[0]):
            prob = self.forward_prop(x[i,:])
            output[i] = np.argmax(prob) 
        return output 

    def repot_error(self, yhat, y):
        # TODO: write a function that compares categorized data to actual output
        print("verify accuracy")

    def write_network_values(self, filename):
        pickle.dump(self, open(filename, "wb"))
        print("Network written to: %s" %(filename))

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print("Layer %d"%(i))
            for j, neuron in enumerate(layer.neurons):
                print("Neuron %d"%(j))
                for k, weight in enumerate(neuron.current_weight):
                    print("%d: %.3f"%(k,weight))
            print("\n")

    def set_initial_conditions(self):
         # create testing initial parameters
         self.layers[0].neurons[0].current_weight[0] = 0.35
         self.layers[0].neurons[0].current_weight[1] = 0.15
         self.layers[0].neurons[0].current_weight[2] = 0.20

         self.layers[0].neurons[1].current_weight[0] = 0.35
         self.layers[0].neurons[1].current_weight[1] = 0.25
         self.layers[0].neurons[1].current_weight[2] = 0.30
         
         self.layers[1].neurons[0].current_weight[0] = 0.60
         self.layers[1].neurons[0].current_weight[1] = 0.40
         self.layers[1].neurons[0].current_weight[2] = 0.45
        
         self.layers[1].neurons[1].current_weight[0] = 0.60
         self.layers[1].neurons[1].current_weight[1] = 0.50
         self.layers[1].neurons[1].current_weight[2] = 0.55

    def set_initial_conditions_3(self):
         # create testing initial parameters
         self.layers[0].neurons[0].current_weight[0] = 1
         self.layers[0].neurons[0].current_weight[1] = 0.1
         self.layers[0].neurons[0].current_weight[2] = 0.2
         self.layers[0].neurons[0].current_weight[3] = 0.3

         self.layers[0].neurons[1].current_weight[0] = 1
         self.layers[0].neurons[1].current_weight[1] = 0.3
         self.layers[0].neurons[1].current_weight[2] = 0.2
         self.layers[0].neurons[1].current_weight[3] = 0.7
         
         self.layers[0].neurons[2].current_weight[0] = 1
         self.layers[0].neurons[2].current_weight[1] = 0.4
         self.layers[0].neurons[2].current_weight[2] = 0.3
         self.layers[0].neurons[2].current_weight[3] = 0.9
         
         self.layers[1].neurons[0].current_weight[0] = 1
         self.layers[1].neurons[0].current_weight[1] = 0.2
         self.layers[1].neurons[0].current_weight[2] = 0.3
         self.layers[1].neurons[0].current_weight[3] = 0.5

         self.layers[1].neurons[1].current_weight[0] = 1
         self.layers[1].neurons[1].current_weight[1] = 0.3
         self.layers[1].neurons[1].current_weight[2] = 0.5
         self.layers[1].neurons[1].current_weight[3] = 0.7
         
         self.layers[1].neurons[2].current_weight[0] = 1
         self.layers[1].neurons[2].current_weight[1] = 0.6
         self.layers[1].neurons[2].current_weight[2] = 0.4
         self.layers[1].neurons[2].current_weight[3] = 0.8

         self.layers[2].neurons[0].current_weight[0] = 1
         self.layers[2].neurons[0].current_weight[1] = 0.1
         self.layers[2].neurons[0].current_weight[2] = 0.4
         self.layers[2].neurons[0].current_weight[3] = 0.8

         self.layers[2].neurons[1].current_weight[0] = 1
         self.layers[2].neurons[1].current_weight[1] = 0.3
         self.layers[2].neurons[1].current_weight[2] = 0.7
         self.layers[2].neurons[1].current_weight[3] = 0.2
         
         self.layers[2].neurons[2].current_weight[0] = 1
         self.layers[2].neurons[2].current_weight[1] = 0.5
         self.layers[2].neurons[2].current_weight[2] = 0.2
         self.layers[2].neurons[2].current_weight[3] = 0.9
class layer:
    def __init__(self,num_inputs,num_neurons, activation):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.sigma = None
        self.activation = activation
        self.neurons = [neuron(self.num_inputs,self.sigma) 
                for i in range(self.num_neurons)]

    def find_neuron_derivatives(self):
        for neuron in self.neurons:
            neuron.derivative = self.activation.derivative(neuron.output)

class neuron:
    def __init__(self,num_inputs,sigma):
        self.output = 0 
        self.weight_der = [None]*(num_inputs+1)  
        self.weight_der = [None]*(num_inputs+1)  
        self.current_weight = [None]*(num_inputs+1) #np.random.normal(0,sigma,[num_inputs+1,1])
        self.new_weight = [None]*(num_inputs+1)
        self.derivative = None

if __name__ == '__main__':
  main()
