"""
Implementing a neural network using the defined layers, optimizers and
other objects

@author: Angel Villar-Corrales
"""


class Neural_Network:
    """
    Object which implements a Neural Network by sequentially calling the 
    forward and backward methods of each layer included.
    """

    def __init__(self, loss_function=None optimizer=None, 
                 weights_initializer=None, bias_initializer=None):
        """
        Initializer of the neural network object

        Args:
        -----
        loss_function: Loss object
            loss function used to compute the error
        optimizer: Optimizer object
            optimizer used to update the network parameters
        weights_initializer: Initializer object
            object that initializes the weights
        bias_initializer: Initializer object
            object that initializes the bias
        """

        # network parameters
        self.layers = []
        self.loss_function = loss_function

        # auxiliary objects: optimizer, initializers, ...
        self.optimizer = optimizer
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
        
        loss = loss_function.predict(predictions=forwarded_tensor, targets=targets)
        self.loss = loss

        return loss


    def backward(self):
        """
        Computing the backward pass through the neural network by sequentially calling the
        backward method of each of the layers
        """

        error = loss_function.backward()
        for layer in layers[::-1]:
            error = layer.backward(error)

        return
