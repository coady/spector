# distutils: language=c++
import collections
import numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

dtype = 'i{}'.format(sizeof(Py_ssize_t))


cdef class indices:
    """A sparse boolean array, i.e., set of indices.

    Provides a memory efficient set interface, with optimized conversion between numpy arrays.

    :param keys: optional iterable of keys
    """
    cdef unordered_set[Py_ssize_t] data

    def __init__(self, keys=()):
        self.update(keys)

    def __repr__(self):
        return 'indices({})'.format(np.array(self))

    def __len__(self):
        return self.data.size()

    def __contains__(self, Py_ssize_t key):
        return self.data.count(key)

    def __iter__(self):
        for k in self.data:
            yield k

    def add(self, Py_ssize_t key):
        return self.data.insert(key).second

    def discard(self, Py_ssize_t key):
        return self.data.erase(key)

    def clear(self):
        self.data.clear()

    def __array__(self):
        """Return keys as numpy array."""
        result = np.empty(len(self), dtype)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        for k in self.data:
            arr[i] = k
            i += 1
        return result

    cdef fromindices(self, indices other):
        for k in other.data:
            self.data.insert(k)

    cdef fromarray(self, Py_ssize_t [:] keys):
        cdef Py_ssize_t i
        for i in range(keys.size):
            self.data.insert(keys[i])

    def update(self, keys):
        """Update from indices or keys."""
        if isinstance(keys, indices):
            return self.fromindices(keys)
        if not isinstance(keys, np.ndarray):
            keys = np.fromiter(keys, dtype)
        self.fromarray(keys)

    @classmethod
    def fromdense(cls, values):
        """Return indices from a dense array representation."""
        return cls(*np.nonzero(values))

    def todense(self, size=None):
        """Return a dense array representation of indices."""
        keys = np.array(self)
        result = np.zeros(keys.max() + 1 if size is None else size, bool)
        result[keys] = True
        return result


cdef class vector:
    """A sparse array of index keys mapped to numeric values.

    Provides a memory efficient dict interface, with optimized conversion between numpy arrays.

    :param keys: optional iterable of keys
    :param values: optional scalar or iterable of values
    :param dtype: optional dtype inferred from values
    """
    cdef public object dtype
    cdef unordered_map[Py_ssize_t, double] data

    def __init__(self, keys=(), values=1, dtype=None):
        self.dtype = np.dtype(dtype or getattr(values, 'dtype', type(values)))
        self.update(keys, values)

    def __repr__(self):
        return 'vector({}, {})'.format(self.keys(), self.values())

    def __len__(self):
        return self.data.size()

    def __getitem__(self, Py_ssize_t key):
        return self.dtype.type(self.data[key])

    def __setitem__(self, Py_ssize_t key, value):
        self.data[key] = value

    def __delitem__(self, Py_ssize_t key):
        self.data.erase(key)

    def __contains__(self, Py_ssize_t key):
        return self.data.count(key)

    def __iter__(self):
        for p in self.data:
            yield p.first

    def items(self):
        return zip(self.keys(), self.values())

    def clear(self):
        self.data.clear()

    def keys(self):
        """Return keys as numpy array."""
        result = np.empty(len(self), dtype)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        for p in self.data:
            arr[i] = p.first
            i += 1
        return result

    def values(self):
        """Return values as numpy array."""
        result = np.empty(len(self), float)
        cdef double [:] arr = result
        cdef Py_ssize_t i = 0
        for p in self.data:
            arr[i] = p.second
            i += 1
        return result.astype(self.dtype)

    cdef fromvector(self, vector other):
        for p in other.data:
            self.data[p.first] = p.second

    cdef fromarrays(self, Py_ssize_t [:] keys, double [:] values):
        cdef Py_ssize_t i
        for i in range(min(keys.size, values.size)):
            self.data[keys[i]] = values[i]

    def update(self, keys, values=1):
        """Update from vector, keys and scalar value, or keys and values."""
        if isinstance(keys, vector):
            return self.fromvector(keys)
        if not isinstance(keys, np.ndarray):
            keys = np.fromiter(keys, dtype)
        if not isinstance(values, collections.Iterable):
            values = np.full(len(keys), values, self.dtype)
        if not isinstance(values, np.ndarray):
            values = np.fromiter(values, self.dtype)
        self.fromarrays(keys, values.astype(float))

    @classmethod
    def fromdense(cls, values):
        """Return vector from a dense array representation."""
        keys, = np.nonzero(values)
        return cls(keys, values[keys])

    def todense(self, size=None):
        """Return a dense array representation of vector."""
        keys = self.keys()
        result = np.zeros(keys.max() + 1 if size is None else size, self.dtype)
        result[keys] = self.values()
        return result
