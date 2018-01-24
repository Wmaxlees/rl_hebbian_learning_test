#!/usr/bin/env python3

import numpy as np

from node import Node

'''
TODO:
    Turn seperate node vectors into a single matrix
    and then use indexing to pull out the appropriate
    parts.

    Then use np.sum() to generate the actual output.

    Then I can save the model, too, using np.save()
'''

class Graph (object):
    def __init__ (self, order):
        print('Generating graph of size %d.' % (order, ))
        self._nodes = np.random.normal(size=(order, order, ), scale=1/order)
        print('Generation complete.')


    def __call__ (self, x, φ):
        """ Feed a value into the network.

        Feed the set of activations x into the network
        and produce the next output step. Any activation
        below φ will be ignored.

        Args:
            x (numpy.ndarray): The input to the network.
            φ (float):         The activation cutoff.

        Returns:
            numpy.ndarray: The new output from the network.
        """
        # Reduce by φ and take the max between x and 0
        x = np.maximum(x - φ, 0)

        # Get all nonzero values
        x = np.nonzero(x)

        # Generate the bit array for weight updating
        bit_array = np.zeros(len(self._nodes))
        bit_array[x] = 1

        # Calculate the new output then update weights
        output = np.sum(self._nodes[x], axis=1)
        self._nodes[x] += bit_array * 0.1

        # Degrade all weights
        self._nodes = self._nodes - 0.03

        self._truncate_weights()

        # Return the average activation
        return output / len(x)

    def predict (self, x, φ):
        # Reduce by φ and take the max between x and 0
        x = np.maximum(x - φ, 0)

        # Get all nonzero values
        x = np.nonzero(x)

        # Calculate the new output
        output = np.sum(self._nodes[x], axis=0)

        # Return the average activation
        return output / len(x)


    def load (self, name):
        self._nodes = np.load(name)


    def save (self, name):
        np.save(name, self._nodes)


    def _truncate_weights (self):
        self._weights = np.minimum(np.maximum(self._nodes, 0), 1)


if __name__ == '__main__':
    raise Warning('graph.py is a module.')
