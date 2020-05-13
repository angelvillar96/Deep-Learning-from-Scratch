"""
Different optimizers used to update the model parateres

@author: Angel Villar-Corrales
"""

import numpy as np


class Optimizer():
    """
    Abstract class for optimizers. 
    Initializes the following methods: 
        __init__(), update_weights() and set_regularizer()
    """

    def __init__(self, learning_rate=1e-3):
        """
        Initializer of the optimizer

        Args:
        -----
        learning_rate: float
            initial learning rate applied in the optimizer
        """

        self.regularization = None
        self.learning_rate = learning_rate

        return


    def update_weights(self, weights, gradients=None):
        """
        Updating the network parameters
        
        Args:
        -----
        weights: numpy array
            network parameters
        gradients: numpy array
            Parameter gradients computed during the backward pass
        
        Returns:
        --------
        updated_weights: numpy array
            updated network parameters
        """

        update_weights = np.copy(weights)

        return update_weights


    def set_regularization(self, regularization):
        """
        Adding regularization to the optimizer

        Args:
        -----
        regularizer: Regularization object
            Regularization method applied to the loss function
        """

        self.regularization = regularization

        return

    
    def regularization_contribution(self, weights):
        """
        Obtaining the update contribution from the regularizer
        
        Args:
        -----
        weights: numpy array
            network parameters

        Returns:
        --------
        regularization_term: numpy array
            contribution to the parameter updates due to regularization
        """

        if(self.regularization != None):
            regularization_term = self.regularization.calculate(weights)
        else:
            regularization_term = np.zeros(weights.shape)
        
        return regularization_term


class Sgd(Optimizer):
    """
    Implementation of Stochastic Gradient Descent optimizer
    """

    def __init__(self, learning_rate=1e-3):
        """
        Initializer of SGD

        Args:
         -----
        learning_rate: float
            initial learning rate applied in the optimizer
        """

        Optimizer.__init__(self)

        self.learning_rate = learning_rate

        return


    def update_weights(self, weights, gradients=None):
        """
        Updating the network parameters
        
        Args:
        -----
        weights: numpy array
            network parameter
        gradients: numpy array
            Parameter gradients computed during the backward pass
        
        Returns:
        --------
        updated_weights: numpy array
            updated network parameters
        """

        update_weights = weights - self.learning_rate * gradients
        regularization_contribution = self.regularization_contribution(weights)
        update_weights = update_weights - self.learning_rate * regularization_contribution

        return update_weights
