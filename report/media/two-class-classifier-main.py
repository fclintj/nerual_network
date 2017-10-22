    num_inputs = 2
    num_outputs= 2 
    batch_size = 200
    epics = 800

    # X,Y = pickle.load(open("./in_out.p","rb"))
    # X,Y,X_test,Y_test = get_classasgn_80_20() 
    X,Y = get_moon_class_data() 
    X_test,Y_test = get_moon_gendata() 
    # X,Y = get_mnist_train("./data")
    
    relu = activation_function(relu_func,relu_der)
    sig  = activation_function(sigmoid_func,sigmoid_der)
    no_activation = activation_function(return_value,return_value)

    num_neurons = 5 
    # input layer
    layers = [layer(num_inputs,num_neurons,sig)]
    layers.append(layer(num_neurons,num_outputs,sig))

    # create neural network
    network = NeuralNetwork(layers) 

    # train network
    network.train_network(X,Y,batch_size,epics)

    # classify data
    Yhat = network.classify_data(X_test)
    network.validate_results(Yhat,Y_test) 
    
    plot_data(X[0:100],X[100:200])
    xtot = np.r_[X,X_test] 
    xlim = [np.min(xtot[:,0]),np.max(xtot[:,0])]
    ylim = [np.min(xtot[:,1]),np.max(xtot[:,1])]
    plot_boundaries(xlim,ylim,network.classify_data)
    plt.show()
    
    # plot error
    network.plot_error()    
    plt.show()

