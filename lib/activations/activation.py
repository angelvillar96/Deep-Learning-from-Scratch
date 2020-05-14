"""
Abstract class for all activations

@author: Angel Villar-Corrales
"""

import numpy as np


class Activation:
    """
    Abstract base class from which all activations inherit
    """

    def __init__(self):
        """
        Intializer of the activation
        """

        return


    def __src__(self):
        """
         Method used to print the activation in a human readable way
        """

        returned_message = f"{self.__class__.__name__}()"

        return returned_message
