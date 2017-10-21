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
    num_inputs = 2
    num_outputs = 2
    batch_size = 200
    epochs = 5000

    # X,Y = pickle.load(open("./in_out.p","rb"))
    # X,Y = get_moon_class_data() 
    
    relu = activation_function(relu_func,relu_der)
    sig  = activation_function(sigmoid_func,sigmoid_der)
    no_activation = activation_function(return_value,return_value)
    
    num_neurons = 23
    # input layer
    layers = [layer(num_inputs,20,sig)]
    layers.append(layer(20,num_neurons,sig))
    layers.append(layer(num_neurons,num_outputs,no_activation))

    # create neural network
    network = NeuralNetwork(layers) 
    network.set_initial_conditions()

    # train network
    network.train_network(X,Y,batch_size,epochs)

    # classify data
    Yhat = network.classify_data(X)
    # print(Yhat)
    # print(Y)
    network.validate_results(Yhat,Y) 
    # print(Yhat)
    # network.write_network_values("network_first.p")

    # plot error
    network.plot_error()    

def get_2_class_data():
    X = np.array([[0.05, 0.1],
                  [0.07, 0.1],
                  [0.05, 0.1],
                  [0.05, 0.1],
                  [0.05, 0.1]])

    Y = np.array([[0.01, 0.99],
                  [0.01, 0.99],
                  [0.01, 0.99],
                  [0.01, 0.99],
                  [0.01, 0.99]])
    return X,Y

def get_3_class_data():
    X = np.array([[0.05, 0.1],
                  [0.07, 0.3],
                  [0.09, 0.5],
                  [0.05, 0.1]])

    Y = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    return X,Y

def get_sprial_class_data():
    np.random.seed(0)
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D))
    y = np.zeros(N*K, dtype='uint8')
    for j in xrange(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    # fig = plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.xlim([-1,1])
    # plt.ylim([-1,1])
    Y = (np.arange(np.max(y) + 1) == y[:, None]).astype(float)
    return X,Y

def get_moon_class_data():
    data = np.loadtxt("./data/classasgntrain1.dat",dtype=float)
    x0 = data[:,0:2]
    x1 = data[:,2:4]
    data = data_frame(x0,x1)
    return data.xtot,data.class_tot

class NeuralNetwork:
    def __init__(self,    layers,   softmax=True,  momentum=0.9, \
                 eta=0.1, scale=0.01  ):

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
        if epochs < 100:
            print_tenth = 1
        percent_complete = 0
        print("Training Data...")
        for i in range(epochs):
            # batch = np.random.randint(0,X.shape[0],batch_size)
            batch = [range(4)]
            for j, sample in enumerate(batch):
                self.error_array = [] 
                # self.train_data(X[sample],Y[sample]) 
                self.train_data(X,Y) 
                if j%MSE_freq is 0:
                    self.total_error.append(np.mean(self.error_array))
                    self.error_array = []
            if i%print_tenth is 0 :
                percent_complete += 1
                print("%d%% MSE: %f"%(percent_complete, self.total_error[i]))
        # print("Total Mean Squared Error: %f"%(np.mean(self.error_array)))

    def train_data(self, X, Y):
        Yhat = self.forward_prop(X)
        dE_dH = (Yhat-Y).T
        iterlayers = iter(self.layers[::-1])

        # back propagation
        if self.softmax is True:
            dE_dWeight = -np.dot((Y-Yhat).T,self.layers[-1].weight_der) / \
                        self.layers[-1].weight_der.shape[0] #+ \
                        # self.scale * self.layers[-1].W

            self.layers[-1].W += -self.eta*(dE_dWeight + self.momentum*self.layers[-1].momentum_matrix)
            self.layers[-1].momentum_matrix = dE_dWeight
            dE_dH = (Yhat-(Y==1).astype(int)).T[0,:]/Yhat.shape[0]
            next(iterlayers)

        for layer in iterlayers:
            dE_dNet = layer.der(layer.output).T*dE_dH
            dE_dWeight = (np.dot(dE_dNet,layer.weight_der))/layer.weight_der.shape[0]
            dE_dH = np.dot(layer.W[:,0].T,dE_dNet)

            layer.momentum_matrix = \
                    self.momentum * layer.momentum_matrix + \
                    self.eta * dE_dWeight
            layer.W += - layer.momentum_matrix

        # self.error_array.append(-np.mean(np.sum(np.log(Yhat)*Y)))
        self.error_array.append(np.mean(sum((Yhat-Y).T*(Yhat-Y).T)))

    def stable_softmax(self, X):
        exp_norm = np.exp(X - np.max(X))
        return exp_norm / np.sum(exp_norm, axis=1).reshape((-1,1))

    def plot_error(self):
        plt.plot(range(len(self.total_error)), self.total_error)
        plt.show()

    def write_network_values(self, filename):
        pickle.dump(self, open(filename, "we"))
        print("Network written to: %s" %(filename))

    def validate_results(self, Yhat, Y):
        Yhat_enc = (np.arange(Y.shape[1]) == Yhat[:, None]).astype(float)
        num_err = np.sum(abs(Yhat_enc - Y))/2
        print("%d Mistakes. Training Accuracy: %.2f%%"%(int(num_err),
            (len(Yhat)-num_err)/len(Yhat)*100))

    def set_initial_conditions(self):
        # self.layers[0].W[0,:] = [0.15,0.2,0.35]
        # self.layers[0].W[1,:] = [0.25,0.3,0.35]
        # self.layers[0].W[2,:] = [0.25,0.3,0.35]

        self.layers[0].W[0,:] = [0.1,0.1,0.01]
        self.layers[0].W[1,:] = [0.2,0.2,0.1 ]
        self.layers[0].W[2,:] = [0.3,0.3,0.1 ]
        
class layer:
    def __init__(self,num_inputs,num_neurons, activation):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = None
        self.weight_der = None
        self.activation = activation
        self.net = None
        self.W = np.random.uniform(0,1,[num_neurons,num_inputs+1])
        self.momentum_matrix = np.zeros([num_neurons,num_inputs+1])
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
