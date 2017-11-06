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
    epochs = 30
    mse_freq = 50

    # open mnist data
    X,Y,X_test,Y_test = get_mnist_train("./data")

    # initialize activation functions
    relu = activation_function(relu_func,relu_der)
    sig  = activation_function(sigmoid_func,sigmoid_der)
    no_activation = activation_function(return_value,return_value)

    num_neurons = 300
    # first layer tests
    layers0 = [layer(num_inputs,num_neurons,relu)]
    layers0.append(layer(num_neurons,num_outputs,no_activation))

    # second layer tests 
    layers1 = [layer(num_inputs,300,relu)]
    layers1.append(layer(300,100,relu))
    layers1.append(layer(100,num_outputs,no_activation))

    # set up test bench
    layer_testbench = [layers0]
    message = ["Layer with 1 hidden network (300 neurons). Epochs " +  "\n",
                "\nLayer with 2 hidden networks (300 and 100 neurons respectively).\n"]

    momentum_values = [0.0,0.8]
    step_size = [0.4,0.7,0.9]

    file = open('../report/media/mnist/test/ten-long-class-network_statistics-bat-'
            + str(batch_size) +
            '-mse-' + str(mse_freq) + '.txt',"w")

    for index, layers in enumerate(layer_testbench):
        file.write(message[index])
        for mom in momentum_values:
            for step in step_size:
                print("Currently on layer " + str(index) + " momentum " + str(mom) + " step size " + str(step))

                # create neural network
                network = NeuralNetwork(layers,eta=step,momentum=mom)

                # train network
                start_time = time.time()
                network.train_network(X,Y,batch_size=batch_size,
                                      epochs=epochs,MSE_freq=mse_freq)
                end_time = time.time()

                # classify data
                Yhat = network.classify_data(X)
                Yhat_test = network.classify_data(X_test)
                training_accuracy = network.validate_results(Yhat,Y)
                training_accuracy = network.validate_results(Yhat_test,Y_test)

                # write statistics
                file.write('mo-' + str(mom) + '-eta-' + str(step) + "\n")
                file.write("Percent Correct: " + str(training_accuracy) + "%\n")
                file.write("Run-time: " + str(end_time-start_time) +" seconds" + "\n\n")

                # plot error
                network.plot_error(index,mom,step)

            # save combined error plot
            plt.title("MSE for Momentum= " + str(mom) +
                      ", Step=" + str(step_size))
            plt.savefig('../report/media/mnist/test/ten-c-bat-' + str(batch_size) +
                      '-mse-' + str(mse_freq) + '-lay-' + str(index) +
                      '-mo-' + str(int(mom*10)) + '-eta-' + str(int(step*10)) +
                      '.pdf',bbox_inches='tight')
            plt.clf()


class NeuralNetwork:
    def __init__(self, layers, softmax=True, momentum=0,
                 eta=0.1, MSE_freq=50, reg=0.001):
        self.num_layers  = len(layers)
        self.num_outputs = layers[self.num_layers-1].num_neurons
        self.error_array = []
        self.error_plot  = []
        self.momentum = momentum
        self.MSE_freq = MSE_freq
        self.softmax= softmax
        self.layers = layers
        self.reg = reg
        self.eta = eta
        self.__set_GRV_starting_weights()

    def train_network(self, X, Y, batch_size=100, epochs=100, MSE_freq=50):
        self.MSE_freq = MSE_freq * batch_size
        print("Training Data...")

        # definition of iterations with mini-batch = N/B*epochs
        itrs_per_epoch = int(np.ceil(X.shape[0]/float(batch_size)))
        total_itrs = itrs_per_epoch * epochs

        # print out 100 samples to gauge speed of program
        if total_itrs > 5000:
            print_frequency = total_itrs/100

        # if iterations are few, print out 10
        else:
            print_frequency = total_itrs/10
            if print_frequency is 0:
                print_frequency += 1 # to avoid modulo by zero

        completed_epocs = 0
        for i in range(total_itrs):
            # randomly select samples from input data for batch
            batch = np.random.randint(0,X.shape[0],batch_size)
            self.train_data(X[batch],Y[batch])
            if i%itrs_per_epoch is 0:
                print("Epoch %d. MSE: %f"%(completed_epocs,
                    np.mean(self.error_array[-self.MSE_freq:])))
                completed_epocs += 1

            if i%print_frequency is 0:
                print("Iteration %d MSE: %f"%(i+1,
                    np.mean(self.error_array[-self.MSE_freq:])))


        # create error plot
        print("Final MSE: %f"%(np.mean(self.error_array[-self.MSE_freq:])))

        # reverse order of list and split into even parts sizeof=MSE_freq 
        plot = self.error_array[::-1]
        for i in range(0,len(plot),self.MSE_freq):
            self.error_plot.append(np.mean(plot[i:i+self.MSE_freq]))
        self.error_plot = self.error_plot[::-1]

    def train_data(self, X, Y):
        Yhat = self.forward_prop(X)
        dE_dH = (Yhat-Y).T

        # back propagation
        if self.softmax is True:
            self.layers[-1].output = np.ones(dE_dH.shape).T

        for layer in self.layers[::-1]:
            dE_dNet = layer.der(layer.output).T*dE_dH

            # divide by number of samples in batch to regularize step
            dE_dWeight = (np.dot(dE_dNet,layer.weight_der)) / \
                of layer.weight_der.shape[0]

            # obtain multiplication to pass back to next layer
            dE_dH = np.dot(layer.W[:,0:-1].T,dE_dNet) * self.reg

            # update weight matrix with momentum
            layer.W += -layer.momentum_matrix
            layer.momentum_matrix = \
                    self.momentum * layer.momentum_matrix + \
                    self.eta * dE_dWeight

        # create long entire list of errors to plot at specific frequency later
        for indx,yhat in enumerate(Yhat):
            self.error_array.append(sum((Y[indx]-yhat)*(Y[indx]-yhat)))

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
        # returns list instead of matrix
        return class_type

    def stable_softmax(self, X):
        exp_norm = np.exp(X - np.max(X))
        return exp_norm / np.sum(exp_norm, axis=1).reshape((-1,1))

    def validate_results(self, Yhat, Y):
        Yhat_enc = (np.arange(Y.shape[1]) == Yhat[:, None]).astype(float)
        num_err = np.sum(abs(Yhat_enc - Y))/2
        training_accuracy = (len(Yhat)-num_err)/len(Yhat)*100
        print("%d Mistakes. Training Accuracy: %.2f%%"%(int(num_err),training_accuracy))
        return training_accuracy

    def plot_error(self,index,momentum,eta):
        plt.plot(range(len(self.error_plot)), self.error_plot)
        plt.xlabel("Mean set for %d Training Samples"%(self.MSE_freq))
        plt.ylabel("Mean Squared Error")

    def write_network_values(self, filename):
        pickle.dump(self, open(filename, "we"))
        print("Network written to: %s" %(filename))

    def __set_GRV_starting_weights(self):
        # find number of outputs at each layer
        for i in range(self.num_layers-2):
            self.layers[i].num_outputs = self.layers[i+1].num_neurons
        self.layers[-1].num_outputs = self.num_outputs

        for layer in self.layers:
            sigma = np.sqrt(float(2) / (layer.num_inputs + layer.num_neurons))
            layer.W = np.random.normal(0,sigma,layer.W.shape)

class layer:
    def __init__(self,num_inputs,num_neurons, activation):
        self.W = np.random.uniform(0,1,[num_neurons,num_inputs+1])
        self.momentum_matrix = np.zeros([num_neurons,num_inputs+1])
        self.num_neurons = num_neurons
        self.num_inputs  = num_inputs
        self.activation  = activation
        self.num_outputs = None
        self.weight_der  = None
        self.net    = None
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

def print_digits(X,ordered,m,n):
    f, ax = plt.subplots(m,n)
    ordered = get_ordered(X);
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

def relu_func(X):
    return np.maximum(0,X)

def relu_der(X):
    X[X<0]=0
    return X

def return_value(X):
    return X

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

def get_ordered_digits(X_train):
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

def get_moon_class_data():
    data = np.loadtxt("./data/classasgntrain1.dat",dtype=float)
    x0 = data[:,0:2]
    x1 = data[:,2:4]
    data = data_frame(x0,x1)
    return data.xtot,data.class_tot

def get_moon_gendata():
    x0 = gendata2(0,10000)
    x1 = gendata2(1,10000)
    data = data_frame(x0,x1)
    return data.xtot, data.class_tot 

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
        
        # create a training set from the classasgntrain1.dat (80% and 20%)
        self.train_x0 = data0[0:80]
        self.train_x1 = data1[0:80]
        self.train_tot = np.r_[data0[0:80],data1[0:80]]
        self.train_class_tot = np.r_[self.class_tot[0:80],self.class_tot[100:180]]
        self.test_data = np.r_[data0[80:100],data1[80:100]]
        self.test_class_tot = np.r_[self.class_tot[80:100],self.class_tot[180:200]]

def get_classasgn_80_20():
    data = np.loadtxt("./data/classasgntrain1.dat",dtype=float)
    x0 = data[:,0:2]
    x1 = data[:,2:4]
    data = data_frame(x0,x1)
    return data.train_tot,data.train_class_tot,data.test_data,data.test_class_tot

def get_mnist_train(file_path):
    mnist = input_data.read_data_sets(file_path)
    X = mnist.train.images
    y = mnist.train.labels.astype("int")
    Y = (np.arange(np.max(y) + 1) == y[:, None]).astype(float)
    X_test = mnist.test.images
    y_test = mnist.test.labels.astype("int")
    Y_test = (np.arange(np.max(y_test) + 1) == y_test[:, None]).astype(float)
    return X,Y,X_test,Y_test

def plot_data(x0,x1):
    xtot = np.r_[x0,x1]
    xlim = [np.min(xtot[:,0]),np.max(xtot[:,0])]
    ylim = [np.min(xtot[:,1]),np.max(xtot[:,1])]

    fig = plt.figure() # make handle to save plot 
    plt.scatter(x0[:,0],x0[:,1],c='red',label='$x_0$')
    plt.scatter(x1[:,0],x1[:,1],c='blue',label='$x_1$')
    plt.xlabel('X Coordinate') 
    plt.ylabel('Y Coordinate') 
    plt.title("Neural Network (Two-class Boundary)")
    plt.legend()

def plot_boundaries(xlim, ylim, equation):
    xp1 = np.linspace(xlim[0],xlim[1], num=100)
    yp1 = np.linspace(ylim[0],ylim[1], num=100) 
    
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

    plt.scatter(blue_pts[0,:],blue_pts[1,:],color='blue',s=0.25)
    plt.scatter(red_pts[0,:],red_pts[1,:],color='red',s=0.25)
    plt.xlim(xlim)
    plt.ylim(ylim)

if __name__ == '__main__':
  main()
