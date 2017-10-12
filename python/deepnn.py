import matplotlib.pyplot as plt
import numpy as np
import math
from   tensorflow.examples.tutorials.mnist import input_data

def main():
    num_inputs = 2 
    num_outputs = 2
    batch_size = 9000 
    epochs = 10
    momentum = 0.5 

    # create the sigmoid activation function
    sigmoid = activation_function(sigmoid_func,sigmoid_derivative)
    softmax = activation_function(softmax_func,softmax_func)

    # create layer structure
    layer0 = layer(2,num_inputs,sigmoid)
    layer1 = layer(2,layer0.num_neurons,sigmoid)
    hidden_layers = [layer0, layer1]

    # create neural network framework
    network = neural_network(num_outputs,hidden_layers,softmax,momentum)

    x = [[0.05, 0.1],
         [0.08, 0.2]]

    y = [[0.00, 1 ],
         [0.00, 1 ]]

    network.train_network(x,y,batch_size,epochs)

    # create testing initial parameters
    # network.layers[0].neurons[0].current_weight[0] = 0.35
    # network.layers[0].neurons[0].current_weight[1] = 0.15
    # network.layers[0].neurons[0].current_weight[2] = 0.20
    #
    # network.layers[0].neurons[1].current_weight[0] = 0.35
    # network.layers[0].neurons[1].current_weight[1] = 0.25
    # network.layers[0].neurons[1].current_weight[2] = 0.30
    #
    # network.layers[1].neurons[0].current_weight[0] = 0.60
    # network.layers[1].neurons[0].current_weight[1] = 0.40
    # network.layers[1].neurons[0].current_weight[2] = 0.45
    #
    # network.layers[1].neurons[1].current_weight[0] = 0.60
    # network.layers[1].neurons[1].current_weight[1] = 0.50
    # network.layers[1].neurons[1].current_weight[2] = 0.55


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
    return x*(1-x)

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

    def __set_layer_sigma(self):
        # set initial guesses for hidden layers - 1
        for i in range(len(self.layers)-1):
            self.layers[i].sigma = np.sqrt(float(2) /
            (self.layers[i].num_inputs + self.layers[i+1].num_inputs))
            for j in range(self.layers[i].num_neurons):
                self.layers[i].neurons[j].current_weight = np.random.normal(0,self.layers[i].sigma,[self.layers[i].num_inputs+1,1])

        # set initial guesses for last hidden layer
        for j in range(self.layers[len(self.layers)-1].num_neurons):
            self.layers[len(self.layers)-1].neurons[j].current_weight = np.random.normal(0,self.layers[i].sigma,[self.layers[len(self.layers)-1].num_inputs+1,1])


    def __update_weights(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.current_weight = neuron.new_weight

    def __layer_opperations(self,X,layers):
        for neuron in layers.neurons:
            neuron.weight_der = X
            neuron.net = np.dot(X,neuron.current_weight)
            neuron.output = self.layers[0].activation.function(neuron.net)
        
        output = [layers.neurons[i].output for i in range(layers.num_neurons)]
        return output

    def backward_prop(self,x,yhat,y):
        eta = 1 
        der_out = yhat-y

        # get derivatives for all layers
        for layer in self.layers:
            layer.find_neuron_derivatives()

        # find weights for last layer
        for j in range(self.layers[self.num_layers-1].num_neurons):
            for k in range(len(self.layers[self.num_layers-1].neurons[0].current_weight)):
                der_neur = self.layers[self.num_layers-1].neurons[0].weight_der[k] * self.layers[self.num_layers-1].neurons[j].derivative*der_out[j]

                self.layers[self.num_layers-1].neurons[j].new_weight[k] = self.layers[self.num_layers-1].neurons[j].current_weight[k] - eta * der_neur

        # find derivative of total output error with respect to penultimate layer
        der_err_tot = 0
        for i in range(len(der_out)):
            der_err_tot += self.layers[self.num_layers-1].neurons[i].current_weight[1] * self.layers[self.num_layers-1].neurons[i].derivative*der_out[i]

         
        # find derivative of total output error with respect to weights in hidden layers 
        for i in range(self.num_layers-2,-1,-1):
            for j in range(self.layers[i].num_neurons):
                for k in range(len(self.layers[i].neurons[0].current_weight)):
                    der_neur = self.layers[i].neurons[j].derivative * self.layers[i].neurons[j].weight_der[k] * der_err_tot

                    # update new weight
                    self.layers[i].neurons[j].new_weight[k] = self.layers[i].neurons[j].current_weight[k] - eta * der_neur

            # calculate new total multiplicative derivative for next layer
            der_err_tot *= self.layers[i].neurons[0].weight_der[1]
        self.__update_weights()

    def train_network(self, x, y, batch_size, epochs):
        batch = np.random.randint(0,len(x),batch_size)
        for i in range(epochs):
            for sample in batch:
                self.train_data(x[sample],y[sample]) 
            print("Total error: %f"%(sum(self.total_error)))

    def train_data(self, x, y):
        yhat = np.array(self.forward_prop(x))
        y = np.array(y).reshape((len(y),1))
        self.backward_prop(x,yhat,y)
        self.total_error = 0.5*(yhat - y)*(yhat - y)

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

    def find_neuron_derivatives(self):
        for neuron in self.neurons:
            neuron.derivative = self.activation.derivative(neuron.output)

class neuron:
    def __init__(self,num_inputs,sigma):
        self.output = 0 
        self.net = 0 
        self.weight_der = 0  
        self.current_weight = [None]*(num_inputs+1) #np.random.normal(0,sigma,[num_inputs+1,1])
        self.new_weight = [None]*(num_inputs+1)
        self.derivative = None

if __name__ == '__main__':
  main()
