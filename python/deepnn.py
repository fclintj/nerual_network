from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import math

def get_ordered(X_train):
    ordered 
          = [ 
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
            print(i*n+j) 
            ordered[i*n+j] = ordered[i*n+j].reshape(28,28)
            ax[i][j].imshow(ordered[i*n+j], cmap = plt.cm.binary, interpolation="nearest")
            ax[i][j].axis("off")

    plt.show()

def softmax(x):
    x_exp = [math.exp(i) for i in x]
    sum_x_exp = sum(x_exp)
    softmax = [i / sum_x_exp for i in x_exp]
    return softmax 

mnist = input_data.read_data_sets("./data") # or wherever you want

# to put your data
# X_train = mnist.train.images
# y_train = mnist.train.labels.astype("int")
# X_test = mnist.test.images
# y_test = mnist.test.labels.astype("int")
# print("X_train.shape=",X_train.shape," y_train.shape=",y_train.shape)
# print("X_test.shape=",X_test.shape," y_test.shape=",y_test.shape)
# # plot one of these
# ordered = get_ordered(X_train)
# print_images(ordered,2,5)

z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
x = softmax(z)
plt.plot(z)
plt.show()

plt.plot(x)
plt.show()
