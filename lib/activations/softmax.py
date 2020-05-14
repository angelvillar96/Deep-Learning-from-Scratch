"""
Softmax activation function

@author: Angel Villar-Corrales
"""

import numpy as np

from lib.activations.activation import Activation

class Softmax(Activation):
    """
    Class implementing the Softmaxs activation function, used at the 
    final layer for classification problems
    """

    def __init__(self):
        """
        Initializer of the softmax object
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
            result of applying the softmax to the input
        """

        self.previous_input = input_tensor

        # ensuring numerical stability
        input_tensor_stable = self._enforce_stability(input_tensor)
        
        batch_size = input_tensor.shape[0]
        denominator = np.sum(np.exp(input_tensor_stable), axis=1)
        output = [np.exp(input_tensor_stable[i])/denominator[i] for i in range(batch_size)]
        output_tensor = np.array(output)

        return output_tensor


    def backward(self, input_error):
        """
        Backward pass through the softmax function. 

        Args:
        -----
        input_error: numpy array
            backpropagated error coming from the layer l+1

        Returns:
        --------
        backprop_error: numpy array
            error backpropagated to layer l-1
        """

        backprop_error = input_error

        return backprop_error


    def _enforce_stability(self, input_tensor):
        """
        Due to the exponential nature, large inputs might become numerical unstable
        This method shifts the data by its maximum, thus enforcing numerical stability

        Args:
        -----
        input_tensor: numpy array
            input to the corresponding layer

        Returns:
        --------
        input_tensor_stable: numpy array
            result of max-shifiting the input tensor
        """

        max_value = np.max(input_tensor)
        input_tensor_stable = input_tensor - max_value
        
        return input_tensor_stable
