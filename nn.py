import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.layer_sizes = layer_sizes
        self.parameters = {}  # Dictionary containing weights and biases of the network
        self.L = len(layer_sizes)  # Number of layers
        for l in range(1, self.L):
            self.parameters['W' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l - 1])
            self.parameters['b' + str(l)] = np.zeros((layer_sizes[l], 1))

    def activation(self, x, func='sig'):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :param func: Type of activation function.
        :return: Vector after applying activation function.
        """
        if func == 'sig':
            return 1 / (1 + np.exp(-x))
        if func == 'tanh':
            return np.tanh(x)
        if func == 'ReLU':
            return np.maximum(0, x)

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        a = x  # The input vector
        for l in range(1, self.L):
            # Retrieve parameters of l-th layer
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]

            # Compute next layer
            z_next = a @ W.T + b.T

            # # Passing the layer to activation function
            # if l + 1 == self.L:
            #     act = 'ReLU'  # for the last layer
            # else:
            #     act = 'sig'  # for other layers
            a_next = self.activation(z_next)

            # Go to next layer
            a = a_next

        aL = a  # The output layer
        return aL
