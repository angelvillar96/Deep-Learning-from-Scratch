"""
Implementation of the Cross Entropy Loss Function

@author: Angel Villar-Corrales
"""

import numpy as np

from lib.loss_functions.loss_function import LossFunction 

class CrossEntropyLoss(LossFunction):
    """
    Implementation of the Cross Entropy Loss Function
    """

    def __init__(self):
        """
        Initialization of the loss
        """

        LossFunction.__init__(self)
        
        self.last_predictions = None
        self.last_labels = None

        return

    
    def forward(self, input_tensor, targets):
        """
        Computing the final loss value

        Args:
        -----
        input_tensor: numpy array
            input to the corresponding layer
        targets: numpy array
            vector containing the labels
        
        Returns:
        --------
        loss: float
            Loss value for the current batch
        """

        self.last_predictions = np.copy(input_tensor)
        self.last_labels = targets

        log_values = -np.log(input_tensor)
        ind_loss_values = log_values*targets
        loss = np.sum(lind_loss_values)

        return loss


    def backward(self):
        """
        Computing the error based on the classifications
        
        Returns:
        --------
        error: numpy array
            error between targets and predictions
        """

        error = self.last_predictions - self.last_labels

        return error


