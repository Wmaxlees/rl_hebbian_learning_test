#!/usr/bin/env python3

import numpy as np

class Node (object):
    def __init__ (self, size, scale):
        self._weights = np.random.normal(scale=scale, size=size)
        self._truncate_weights()

    def degrade (self, λ):
        """ Reduce the weights of all nodes.

        Reduces the weights of all nodes by reducing the
        values by some constant.
        """
        self._weights = λ * self._weights
        self._truncate_weights()

    def get_weights (self):
        """ Get the weights for each node in the network.

        Returns:
            numpy.ndarray: The weights for each other node in
                           the graph.
        """
        return self._weights

    def increase_weights (self, bit_array, β):
        """ Increase specific weights of the node.

        Use the bit array to identify which nodes to increase
        and increase their value by β.

        Args:
            bit_array (numpy.ndarray): A binary array of nodes whose
                                       values should be increased.
            β (float):                 The amount to increase each node
        """
        update_value = bit_array * β
        self._weights = self._weights + update_value
        self._truncate_weights()

    def _truncate_weights (self):
        self._weights = np.minimum(np.maximum(self._weights, 0), 1)


if __name__ == '__main__':
    raise Warning('node.py is a module.')
