# Code taken from: Omar Aflak, https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
# Also comes with a very good explanation on how backwards propagation works

import numpy as np

class Layer:
    def forward_propagation(self, input):
        raise NotImplementedError
    
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        # based on their working out, aren't you supposed to divide by len(output_error)?
        input_error = np.dot(output_error, self.weights.T)
        # added np.asmatrix, since unalligned dimensions: "shapes (784,) and (1,100) not aligned"
        weights_error = np.dot(np.asmatrix(self.input).T, output_error)
        bias_error = output_error
        self.weights -= weights_error * learning_rate
        self.bias -= bias_error * learning_rate
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)        

class Network():
    def __init__(self):
        self.loss = None
        self.loss_prime = None
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output
    
    def train(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            total_error = 0.0
            for j in range(samples):
                input = x_train[j]
                y_pred = self.predict(input)
                y_true = y_train[j]

                err = self.loss(y_pred, y_true)
                total_error += err

                error = self.loss_prime(y_pred, y_true)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            print(f"epochs={i+1}/{epochs} error={total_error/samples:.5f}")
    
    def save(self, file_name):
        #numpy.save: https://numpy.org/doc/stable/reference/generated/numpy.save.html 
        with open(file_name, 'wb') as file:
            for layer in self.layers:
                if isinstance(layer, FCLayer):
                    np.save(file, layer.weights)
                    np.save(file, layer.bias)
    
    def load(self, file_name):
        #numpy.load: https://numpy.org/doc/stable/reference/generated/numpy.load.html 
        with open(file_name, 'rb') as file:
            for layer in self.layers:
                if isinstance(layer, FCLayer):
                    layer.weights = np.load(file)
                    layer.bias = np.load(file)
    
