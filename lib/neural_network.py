"""
Implementing a neural network using the defined layers, optimizers and
other objects

@author: Angel Villar-Corrales
"""

from __future__ import print_function

import numpy as np


class Neural_Network:
    """
    Object which implements a Neural Network by sequentially calling the 
    forward and backward methods of each layer included.
    """

    def __init__(self, loss_function=None, optimizer=None, regularization=None,
                 weights_initializer=None, bias_initializer=None, model_name="Neural Network"):
        """
        Initializer of the neural network object

        Args:
        -----
        loss_function: Loss object
            loss function used to compute the error
        optimizer: Optimizer object
            optimizer used to update the network parameters
        regularization: Regularization object
            Regularization method applied to the loss function
        weights_initializer: Initializer object
            object that initializes the weights
        bias_initializer: Initializer object
            object that initializes the bias
        model_name: string
            String used to name the model
        """

        # network parameters
        self.layers = []
        self.loss_function = loss_function

        # auxiliary objects: optimizer, initializers, ...
        self.model_name = model_name
        self.optimizer = optimizer
        self.regularization = regularization
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        # relevant variables
        self.loss_value = None

        return


    def forward(self, input_tensor, targets):
        """
        Forward pass through the neural network. Sequentially calling the forward
        method of each of the layers

        Args:
        -----
        input_tensor: numpy array
            input to the corresponding layer
        targets: numpy array
            labels that the neural network tries to predict
        """

        forwarded_tensor = np.copy(input_tensor)
        for layer in layers:
            forwarded_tensor = layer.forward(forwarded_tensor)
        
        loss = self.loss_function.predict(predictions=forwarded_tensor, targets=targets)
        self.loss = loss

        return loss


    def backward(self):
        """
        Computing the backward pass through the neural network by sequentially calling the
        backward method of each of the layers
        """

        error = self.loss_function.backward()
        for layer in layers[::-1]:
            error = layer.backward(error)

        return


    def add_layer(self, layer):
        """
        Adding a layer to the model, while keeping the previous ones

        Args:
        -----
        layer: Layer Object
            Layer objects to add to the model
        """

        self.layers.append(layer)

        return


    def set_layers(self, layers):
        """
        Setting the layers of the model given the input parameter

        Args:
        -----
        layers: list
            List containing Layer objects
        """

        self.layers = layers

        return


    def set_optimizer(self, optimizer):
        """
        Adding an optimizer to the neural network

        Args:
        -----
        optimizer: Optimizer object
            Optimizer used to update the model paraters (SGD, Adam, ...)
        """

        self.optimizer = optimizer

        return


    def set_loss_function(self, loss_function):
        """
        Adding a loss function to the neural network

        Args:
        -----
        loss_function: Loss Function object
            Loss function used to compute the loss value
        """

        self.loss_function = loss_function

        return


    def set_regularization(self, regularization):
        """
        Adding a regularizer to the neural network

        Args:
        -----
        regularization: Regularization object
            Regularization method applied to the loss function
        """

        if(self.optimizer is None or self.loss_function is None):
            message = "WARNING\nOptimizer and Loss Function must be set befor adding a regularizer"
            assert False, message

        self.regularization = regularization
        self.optimizer.set_regularization(regularization)
        self.loss_function.set_regularization(regularization)
        
        return


    def __str__(self):
        """
        Method used to print the structure of a network
        """

        returned_message = []
        tab = " "*4
        returned_message.append(f"{self.model_name}")
        returned_message.append("-"*len(self.model_name))

        returned_message.append("")
        returned_message.append("Network Attributes:")
        returned_message.append(f"{tab}Loss Function: {self.loss_function.__class__.__name__}")
        returned_message.append(f"{tab}Optimizer: {self.optimizer.__class__.__name__}")
        returned_message.append(f"{tab}Regularizer: {self.regularization.__class__.__name__}")

        returned_message.append("")
        returned_message.append("Layers:")
        for i, l in enumerate(self.layers):
            returned_message.append(f"{tab}{i+1}.  {l.__src__()}")
        
        returned_message.append("")
        returned_message = "\n".join(returned_message)
        
        return returned_message
