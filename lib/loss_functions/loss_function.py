"""
Abstract class for all Loss Functions

@author: Angel Villar-Corrales
"""

import numpy as np


class LossFunction:
    """
    Abstract base class from which all loss functions inherit
    """

    def __init__(self):
        """
        Initializer of the loss function object
        """

        self.regularizer = None

        return


    def __str__(self):
        """
        Method for printing in a human-readable way the loss function
        """

        returned_message = []
        tab = " "*4
        returned_message.append(f"Loss Function: {self.__class__.__name__}")

        #returned_message.append("")
        returned_message = "\n".join(returned_message)

        return returned_message
