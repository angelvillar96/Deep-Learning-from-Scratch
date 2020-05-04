"""
Implementation of a fully connected (FC) layer

@author: Angel Villar-Corrales
"""

import numpy as np


class FullyConnected:
    """
    Class corresponding to a fully connected layer of a neural network

    Args:
    -----
    input_size: integer
        number of inputs connections to the layer
    output_size: integer
        number of output connections from the layer
    """

    def __init__(self, input_size=32, output_size=32):
        """
        Initializer of the FC layer
        """

        # arguments
        self.input_size = input_size
        self.output_size = output_size

        # network parameters 
        self.weight_shape = (input_size+1, output_size)
        self.weights = np.random.rand(*self.weight_shape)
        self.gradients = np.random.rand (*self.weight_shape)

        # auxiliary objects
        self.optimizer = None
        self.previous_input = None

        return


    def forward(self, input_tensor):
        """
        Forward pass through the FC layer

        Args:
        -----
        input_tensor: numpy array
            input to the corresponding layer
        
        Returns:
        --------
        output_tensor: numpy array
            input to the corresponding layer
        """

        bias = np.ones((input_tensor.shape[0], 1), dtype=float)
        input_tensor = np.concatenate((input_tensor, bias), axis=1)
        output_tensor = np.matmul(self.weights.T, input_tensor.T).T

        # saving input for the backward pass
        self.previous_input = input_tensor

        return output_tensor


    def backward(self, input_error):
        """
        Backward pass through the FC layer. Updating layer parameters
        and backpropagating the error

        Args:
        -----
        input_error: numpy array
            backpropagated error coming from the layer l+1

        Returns:
        --------
        backprop_error: numpy array
            error backpropagated to layer l-1
        """

        backprop_error = np.matmul(input_error, self.weights.T)
        backprop_error = backprop_error[:,:-1]  # error due to bias is not backpropagated 

        gradients = np.matmul(self.previous_input.T, input_error)
        if(self.optimizer != None):
            # ToDo
            self.weights = self.optimizer()
        self.gradients = gradients

        return backprop_error


    def _parameter_initializer(self, weights_initializer, bias_initializer):
        """
        Intializer of the layer parameters
        
        Args:
        -----
        weights_initializer: Initializer object
            object that initializes the weights
        bias_initializer: Initializer object
            object that initializes the bias
        """

        weights = weights_initializer.initialize([self.input_size, self.output_size], self.input_size, self.output_size)
        bias = bias_initializer.initialize([1, self.output_size], 1, self.output_size)
        
        self.weights = np.concatenate((weights, bias), axis=0)
        self.weightGradient = np.concatenate((weights, bias), axis=0)

        return
