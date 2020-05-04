"""
Implementation of a fully connected (FC) layer
"""

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