import collections
import numpy as np
from .vector import vector


class matrix(collections.defaultdict):
    """A sparse vector of vectors."""
    def __init__(self, *args, **kwargs):
        super(matrix, self).__init__(vector, *args, **kwargs)

    @property
    def row(self):
        """COO format row index array of the matrix"""
        return np.concatenate([np.full(len(self[key]), key) for key in self])

    @property
    def col(self):
        """COO format column index array of the matrix"""
        return np.concatenate([vec.keys() for vec in self.values()])

    @property
    def data(self):
        """COO format data array of the matrix"""
        return np.concatenate([vec.values() for vec in self.values()])
