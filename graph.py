#!/usr/bin/env python3

import cudamat as cm
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
        cuda_x = cm.CUDAMatrix(np.array([x]))
        cuda_x.add(-φ, target=cuda_x)
        cuda_x.copy_to_host()
        cuda_x.numpy_array = np.maximum(cuda_x.numpy_array[0], 0)

        # Get all nonzero values
        indices = np.nonzero(cuda_x.numpy_array)

        # Generate the bit array for weight updating
        bit_array = np.zeros(len(self._nodes))
        bit_array[indices] = 1

        # Calculate the new output then update weights
        cuda_in = cm.CUDAMatrix(self._nodes[indices])
        cuda_out = cm.empty((1, len(x)))
        cuda_in.sum(axis=0, target=cuda_out)
        cuda_out.mult(len(x))
        output = cuda_out.asarray()[0]
        self._nodes[indices] += bit_array * 0.1

        # Degrade all weights
        self._nodes = self._nodes - 0.03

        self._truncate_weights()

        # Return the average activation
        return output / len(x)

    def predict (self, x, φ):
        # Reduce by φ and take the max between x and 0
        cuda_x = cm.CUDAMatrix(np.array([x]))
        cuda_x.add(-φ, target=cuda_x)
        cuda_x.copy_to_host()
        cuda_x.numpy_array = np.maximum(cuda_x.numpy_array[0], 0)

        # Get all nonzero values
        indices = np.nonzero(cuda_x.numpy_array)

        # Calculate the new output
        cuda_in = cm.CUDAMatrix(self._nodes[indices])
        cuda_out = cm.empty((1, len(x)))
        cuda_in.sum(axis=0, target=cuda_out)
        cuda_out.mult(len(x))
        output = cuda_out.asarray()[0]

        return output


    def load (self, name):
        self._nodes = np.load(name)


    def save (self, name):
        np.save(name, self._nodes)


    def _truncate_weights (self):
        self._weights = np.minimum(np.maximum(self._nodes, 0), 1)


if __name__ == '__main__':
    raise Warning('graph.py is a module.')
