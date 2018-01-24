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
        self._nodes = []
        for idx in range(order):
            if idx % 1000 == 0:
                print('Node %d complete.' % (idx, ))

            self._nodes.append(Node(order, 1/order))
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
        weights = np.zeros(len(self._nodes))
        for node in np.array(self._nodes)[x]:
            weights = weights + node.get_weights()
            node.increase_weights(bit_array, 0.1)

        # Degrade all weights
        for node in self._nodes:
            node.degrade(0.03)

        # Return the average activation
        return weights / len(x)

    def predict (self, x, φ):
        # Reduce by φ and take the max between x and 0
        x = np.maximum(x - φ, 0)

        # Get all nonzero values
        x = np.nonzero(x)


        # Calculate the new output
        weights = np.zeros(len(self._nodes))
        for node in np.array(self._nodes)[x]:
            weights = weights + node.get_weights()

        # Return the average activation
        return weights / len(x)




if __name__ == '__main__':
    raise Warning('graph.py is a module.')
