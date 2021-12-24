import collections
import functools
from typing import Callable, Iterable, Iterator, Mapping
import numpy as np  # type: ignore
from .vector import arggroupby as _arggroupby, asiarray, vector


def arggroupby(values: Iterable) -> Iterator[tuple]:
    """Generate unique keys with corresponding index arrays."""
    values = np.asarray(values)
    keys, counts = np.unique(values, return_counts=True)
    return zip(keys, np.split(np.argsort(values), np.cumsum(counts)))


def groupby(keys: Iterable, *arrays) -> Iterator[tuple]:
    """Generate unique keys with associated groups.

    Args:
        keys:
        *arrays (Iterable):
    """
    arrays = tuple(map(np.asarray, arrays))
    try:
        items = _arggroupby(asiarray(keys))
    except TypeError:  # fallback to sorting
        items = arggroupby(keys)
    for key, values in items:
        yield (key,) + tuple(arr[values] for arr in arrays)


class matrix(collections.defaultdict):
    """A sparse vector of vectors.

    Args:
        data (Iterable):
    """

    def __init__(self, data=(), copy=True):
        super().__init__(vector)
        (self if copy else super()).update(data)

    @classmethod
    def cast(cls, data) -> 'matrix':
        return cls(data, copy=False)

    @property
    def row(self) -> np.ndarray:
        """COO format row index array of the matrix"""
        return np.concatenate([np.full(len(self[key]), key) for key in self])

    @property
    def col(self) -> np.ndarray:
        """COO format column index array of the matrix"""
        return np.concatenate([vec.keys() for vec in self.values()])

    @property
    def data(self) -> np.ndarray:
        """COO format data array of the matrix"""
        return np.concatenate([vec.values() for vec in self.values()])

    def update(self, data):
        """Update from mapping or iterable."""
        if isinstance(data, Mapping):
            for key in data:
                self[key].update(data[key])
        else:
            for key, value in data:
                self[key].update(value)

    def __iadd__(self, other):
        if isinstance(other, Mapping):
            for key in other:
                self[key] += other[key]
        else:
            self.map(vector.__iadd__, other)
        return self

    def __add__(self, other) -> 'matrix':
        return type(self)(self).__iadd__(other)

    def __imul__(self, other):
        if isinstance(other, Mapping):
            for key in set(self).difference(other):
                del self[key]
            for key in self:
                self[key] *= other[key]
        else:
            self.map(vector.__imul__, other)
        return self

    def __mul__(self, other) -> dict:
        if isinstance(other, Mapping):
            return self.cast((key, self[key] * other[key]) for key in set(self).intersection(other))
        return self.map(vector.__mul__, other)

    def sum(self, axis: int = None):
        """Return sum of matrix elements across axis, by default both."""
        if axis in (0, -2):
            return functools.reduce(vector.__iadd__, self.values(), vector())
        if axis in (1, -1):
            return self.map(np.sum)
        if axis is None:
            return sum(map(np.sum, self.values()))
        raise np.AxisError(axis, ndim=2)

    def map(self, func: Callable, *args, **kwargs) -> dict:
        """Return matrix with function applies across vectors."""
        result = {key: func(self[key], *args, **kwargs) for key in self}
        return self.cast(result) if all(isinstance(value, vector) for value in result.values()) else result

    def filter(self, func: Callable, *args, **kwargs) -> 'matrix':
        """Return matrix with function applies across vectors."""
        return self.cast((key, vec) for key, vec in self.items() if func(vec, *args, **kwargs))

    @classmethod
    def fromcoo(cls, row: Iterable, col: Iterable[int], data: Iterable[float]) -> 'matrix':
        """Return matrix from COOrdinate format arrays."""
        return cls.cast((key, vector(col, data)) for key, col, data in groupby(row, col, data))

    def transpose(self) -> 'matrix':
        """Return matrix with reversed dimensions."""
        return self.fromcoo(self.col, self.row, self.data)

    T = property(transpose)

    def __matmul__(self, other: 'matrix') -> 'matrix':
        other = other.transpose()
        return self.cast((key, vector(other.map(self[key].__matmul__))) for key in self)
