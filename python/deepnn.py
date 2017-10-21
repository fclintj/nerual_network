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
    num_inputs = 784
    num_outputs = 10 
    batch_size = 500
    epochs = 70

    mnist = input_data.read_data_sets("/tmp/data/") # or wherever you want

    # to put your data
    X_train = mnist.train.images
    y_train = mnist.train.labels.astype("int")
    y_enc = (np.arange(np.max(y_train) + 1) == y_train[:, None]).astype(float)

    X_test = mnist.test.images
    y_test = mnist.test.labels.astype("int")

    relu = activation_function(relu_func,relu_der)
    sig  = activation_function(sigmoid_func,sigmoid_der)
    no_activation = activation_function(return_value,return_value)
    
    num_neurons = 10
    # input layer
    layers = [layer(num_inputs,num_neurons,sig)]
    # hidden layers
    for i in range(2):
        layers.append(layer(num_neurons,num_neurons,sig))
    # output layer
    layers.append(layer(num_neurons,num_outputs,no_activation))

    # create neural network
    network = NeuralNetwork(layers) 

    # train network
    network.train_network(X_train[0:11],y_enc[0:11],batch_size,epochs)
    
    # classify data
    yhat = network.classify_data(X_train[0:11])
    print(yhat) 
    print(y_train)

    # network.write_network_values("network_first.p")

    # plot error
    network.plot_error()    

class NeuralNetwork:
    def __init__(self,    layers,   softmax=True,  momentum=0, \
                 eta=0.9, scale=0.01  ):

        self.softmax=softmax
        self.num_layers = len(layers)
        self.num_outputs = layers[self.num_layers-1].num_neurons
        self.layers = layers
        self.momentum = momentum
        self.scale = scale
        self.eta = eta 
        self.softmax = softmax 
        self.total_error = [] 
        self.error_array = [] 
        self.__set_GRV_starting_weights()

    def __set_GRV_starting_weights(self):
        for i in range(self.num_layers-2):
            self.layers[i].num_outputs = self.layers[i+1].num_neurons
        self.layers[-1].num_outputs = self.num_outputs

        for layer in self.layers:
            sigma = np.sqrt(float(2) / (layer.num_inputs + layer.num_inputs)) 
            layer.W = np.random.normal(0,sigma,layer.W.shape)

    def forward_prop(self, X):
        prev_out = X
        for layer in self.layers:
            prev_out = np.c_[prev_out,np.ones([prev_out.shape[0],1])]
            prev_out = layer.forward(prev_out)

        if self.softmax is True:
            self.layers[-1].output = self.stable_softmax(self.layers[-1].net)

        return self.layers[-1].output 
         
    def classify_data(self, X):
        Yhat = self.forward_prop(X)
        class_type = np.argmax(Yhat,axis=1)
        return class_type

    def train_network(self, X, Y, batch_size, epochs, MSE_freq=30):
        error_array_output = []
        print_tenth = epochs/100
        percent_complete = 0
        print("Training Data...")
        for i in range(epochs):
            batch = np.random.randint(0,X.shape[0],batch_size)
            for j, sample in enumerate(batch):
                self.error_array = [] 
                self.train_data(X[0:11],Y[0:11]) 
                if j%MSE_freq is 0:
                    self.total_error.append(np.mean(self.error_array))
                    self.error_array = []
            if i%print_tenth is 0 :
                percent_complete += 1
                print("%d%% MSE: %f"%(percent_complete, self.total_error[i]))
        print("Total Mean Squared Error: %f"%(np.mean(self.error_array)))

    def train_data(self, X, Y):
        Yhat = self.forward_prop(X)
        dE_dH = (Yhat-Y).T

        # back propagation
        if self.softmax is True:
            dE_dWeight = -np.dot((Y-Yhat).T,self.layers[-1].weight_der) / \
                        self.layers[-1].weight_der.shape[0] + \
                        self.scale * self.layers[-1].W

            self.layers[-1].momentum_matrix = \
                    self.momentum * self.layers[-1].momentum_matrix + \
                    self.eta * dE_dWeight
            self.layers[-1].W += - self.layers[-1].momentum_matrix
            # dE_dH = (Yhat-(Y==1).astype(int))[0].T/Yhat.shape[0]

        iterlayers = iter(self.layers[::-1])
        next(iterlayers)
        for layer in iterlayers:
            dE_dNet = np.dot(layer.der(layer.output),dE_dH.T) 
            dE_dWeight = (np.dot(dE_dNet,layer.weight_der))/layer.weight_der.shape[0]
            dE_dH = np.dot(layer.W[:,0].T,dE_dNet)

            layer.momentum_matrix = \
                    self.momentum * layer.momentum_matrix + \
                    self.eta * dE_dWeight
            layer.W += - layer.momentum_matrix

        # self.error_array.append(-np.mean(np.sum(np.log(Yhat)*Y)))
        self.error_array.append(np.mean(sum((Yhat-Y).T*(Yhat-Y).T)))

    def stable_softmax(self, Z):
        Z = np.maximum(Z, -1e3)
        Z = np.minimum(Z, 1e3)
        numerator = np.exp(Z)
        denom = np.sum(numerator, axis=1).reshape((-1,1))
        return numerator / denom 

    def plot_error(self):
        plt.plot(range(len(self.total_error)), self.total_error)
        plt.show()

    def write_network_values(self, filename):
        pickle.dump(self, open(filename, "we"))
        print("Network written to: %s" %(filename))

    def set_initial_conditions(self):
        self.layers.W[0,:] = [0.1,0.1,0.01]
        self.layers.W[1,:] = [0.2,0.2,0.1 ]
        self.layers.W[2,:] = [0.3,0.3,0.1 ]
        
class layer:
    def __init__(self,num_inputs,num_neurons, activation):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = None
        self.weight_der = None
        self.activation = activation
        self.net = None
        self.W = np.random.uniform(0,1,[num_neurons,num_inputs+1])
        self.momentum_matrix = np.empty([num_neurons,num_inputs+1])
        self.output = None  
    

    def forward(self, X):
        self.weight_der = X
        self.net = np.dot(X, self.W.T) 
        self.output = self.activation.function(self.net)
        return self.output

    def der(self, X):
        return self.activation.derivative(X)

    def set_initial_conditions(self):
        print("test")

class activation_function:
    def __init__(self,function,derivative):
        self.function = function
        self.derivative = derivative

    def function(self,x):
        return self.function(x) 

    def derivative(self,x):
        return self.derivative(x) 

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

def test_bi_linear(network,data,yhat):
    print(yhat)
    num_err = sum(abs(yhat.reshape(-1,1) - data.y))
    print("Percent of errors: %.4f"%(float(num_err)/data.N))

    test_data = data_frame(gendata2(0,10000),gendata2(1,10000))
    yhat = network.classify_data(test_data.xtot)
    num_err = sum(abs(yhat.reshape(-1,1)- test_data.y))
    print("Percent of errors: %.5f"%(float(num_err)/test_data.N))
    plot_boundaries(data,network.classify_data) 

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return (x*(1-x))

def return_value(X):
    return X

def relu_func(X):
    return np.maximum(0,X)
    
def relu_der(X):
    X[X<0]=0
    return X

def stable_softmax_func(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def softmax_func(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

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
            point = np.array([x,y]).reshape(1,2)
            prob = equation(point)
            if prob == 0:
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

if __name__ == '__main__':
  main()
