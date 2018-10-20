import collections
import functools
import numpy as np
from .vector import arggroupby, vector
try:
    from future_builtins import zip
except ImportError:
    pass


def groupby(keys, *columns):
    """Generate unique keys with matching array slices, similiar to pandas' groupby."""
    keys = np.asarray(keys)
    order = np.argsort(keys)
    columns = [np.asarray(values)[order] for values in columns]
    keys, counts = np.unique(keys[order], return_counts=True)
    start = 0
    for key, stop in zip(keys, counts.cumsum()):
        yield (key,) + tuple(values[start:stop] for values in columns)
        start = stop


class matrix(collections.defaultdict):
    """A sparse vector of vectors."""
    def __init__(self, data=(), copy=True):
        super(matrix, self).__init__(vector)
        if copy:
            self.update(data)
        else:
            super(matrix, self).update(data)

    @classmethod
    def cast(cls, data):
        return cls(data, copy=False)

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
            self.map(vector.__iadd__, other)
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
            self.map(vector.__imul__, other)
        return self

    def __mul__(self, other):
        if isinstance(other, collections.Mapping):
            return self.cast((key, self[key] * other[key]) for key in set(self).intersection(other))
        return self.map(vector.__mul__, other)

    def sum(self, axis=None):
        """Return sum of matrix elements across axis, by default both."""
        if axis in (0, -2):
            return functools.reduce(vector.__iadd__, self.values(), vector())
        if axis in (1, -1):
            return self.map(np.sum)
        if axis is None:
            return sum(map(np.sum, self.values()))
        raise np.AxisError("axis {} is out of bounds".format(axis))

    def map(self, func, *args, **kwargs):
        """Return matrix with function applies across vectors."""
        result = {key: func(self[key], *args, **kwargs) for key in self}
        return self.cast(result) if all(isinstance(value, vector) for value in result.values()) else result

    def filter(self, func, *args, **kwargs):
        """Return matrix with function applies across vectors."""
        return self.cast((key, vec) for key, vec in self.items() if func(vec, *args, **kwargs))

    @classmethod
    def fromcoo(cls, row, col, data):
        """Return matrix from COOrdinate format arrays."""
        row, col, data = map(np.asarray, [row, col, data])
        try:
            row = row.astype(np.intp, casting='safe', copy=False)
        except TypeError:  # fallback to sorting
            return cls.cast((key, vector(col, data)) for key, col, data in groupby(row, col, data))
        return cls.cast((key, vector(col[group], data[group])) for key, group in arggroupby(row))

    def transpose(self):
        """Return matrix with reversed dimensions."""
        return self.fromcoo(self.col, self.row, self.data)

    def __matmul__(self, other):
        other = other.transpose()
        return self.cast((key, vector(other.map(self[key].dot))) for key in self)
