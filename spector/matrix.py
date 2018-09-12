import collections
import functools
import numpy as np
from .vector import vector


class matrix(collections.defaultdict):
    """A sparse vector of vectors."""
    def __init__(self, data=()):
        super(matrix, self).__init__(vector)
        self.update(data)

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

    def update(self, data):
        """Update from mapping or iterable."""
        if isinstance(data, collections.Mapping):
            for key in data:
                self[key].update(data[key])
        else:
            for key, value in data:
                self[key].update(value)

    def __iadd__(self, other):
        if isinstance(other, collections.Mapping):
            for key in other:
                self[key] += other[key]
        else:
            for vec in self.values():
                vec += other
        return self

    def __add__(self, other):
        return type(self)(self).__iadd__(other)

    def __imul__(self, other):
        if isinstance(other, collections.Mapping):
            for key in set(self).difference(other):
                del self[key]
            for key in self:
                self[key] *= other[key]
        else:
            for vec in self.values():
                vec *= other
        return self

    def __mul__(self, other):
        result = type(self)()
        if isinstance(other, collections.Mapping):
            for key in set(self).intersection(other):
                result[key] = self[key] * other[key]
        else:
            for key in self:
                result[key] = self[key] * other
        return result

    def sum(self, axis=None):
        """Return sum of matrix elements across axis, by default both."""
        if axis in (0, -2):
            return functools.reduce(vector.__iadd__, self.values(), vector())
        if axis in (1, -1):
            return {key: np.sum(self[key]) for key in self}
        if axis is None:
            return sum(map(np.sum, self.values()))
        raise np.AxisError("axis {} is out of bounds".format(axis))
