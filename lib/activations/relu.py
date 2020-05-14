"""
ReLU Activation function

@author: Angel Villar-Corrales
"""

import numpy as np

from lib.activations.activation import Activation

class ReLU(Activation):
    """
    Class implementing the Rectified Linear Unit activation function
    """

    def __init__(self):
        """
        Initializer of the relu object
        """

        Activation.__init__(self)
        
        self.previous_input = None

        return


    def forward(self, input_tensor):
        """
        Forward pass the ReLU function

         Args:
        -----
        input_tensor: numpy array
            input to the corresponding layer

        Returns:
        --------
        output_tensor: numpy array
            result of applying the activation to the input
        """

        self.previous_input = input_tensor
        
        mask = np.zeros(input_tensor.shape)
        output_tensor = np.maximum(input_tensor, mask)

        return output_tensor


    def backward(self, input_error):
        """
        Backward pass through the ReLU function. We simply set negatives value to zero,
        since the derivative of the positive stays untouched

        Args:
        -----
        input_error: numpy array
            backpropagated error coming from the layer l+1

        Returns:
        --------
        backprop_error: numpy array
            error backpropagated to layer l-1
        """

        negativ_idx = np.where(self.previous_input <= 0)[0]
        backprop_error = np.copy(input_error)
        backprop_error[negativ_idx] = 0

        return backprop_error

