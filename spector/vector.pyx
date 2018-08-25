# distutils: language=c++
import collections
import numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
cimport cython
try:
    from future_builtins import zip
except ImportError:
    pass

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

    cdef issubset(self, indices other):
        for k in self.data:
            if not other.data.count(k):
                return False
        return True

    def __eq__(self, other):
        return isinstance(other, indices) and len(self) == len(other) and self.issubset(other)

    def __le__(self, indices other):
        return len(self) <= len(other) and self.issubset(other)

    def __lt__(self, indices other):
        return len(self) < len(other) and self.issubset(other)

    def isdisjoint(self, indices other):
        """Return whether two indices have a null intersetction."""
        for k in self.data:
            if other.data.count(k):
                return False
        return True

    def add(self, Py_ssize_t key):
        """Add an index key."""
        return self.data.insert(key).second

    def discard(self, Py_ssize_t key):
        """Remove an index key, if present."""
        return self.data.erase(key)

    def clear(self):
        """Remove all indices."""
        self.data.clear()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __array__(self):
        """Return keys as numpy array."""
        result = np.empty(len(self), dtype)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        for k in self.data:
            arr[i] = k
            i += 1
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef fromarray(self, Py_ssize_t [:] keys):
        cdef Py_ssize_t i
        for i in range(keys.size):
            self.data.insert(keys[i])

    def update(self, keys):
        """Update from indices, array, or iterable."""
        if isinstance(keys, indices):
            self |= keys
        elif isinstance(keys, np.ndarray):
            self.fromarray(keys)
        else:
            for key in keys:
                self.data.insert(key)

    def __ior__(self, indices other):
        for k in other.data:
            self.data.insert(k)
        return self

    def __or__(self, indices other):
        return type(self)(self).__ior__(other)

    def __ixor__(self, indices other):
        for k in other.data:
            if not self.data.insert(k).second:
                self.data.erase(k)
        return self

    def __xor__(self, indices other):
        return type(self)(self).__ixor__(other)

    def __iand__(self, indices other):
        for k in self.data:
            if not other.data.count(k):
                self.data.erase(k)
        return self

    def __and__(indices self, indices other):
        if len(other) < len(self):
            return other & self
        cdef indices result = type(self)()
        for k in self.data:
            if other.data.count(k):
                result.data.insert(k)
        return result

    def __isub__(self, indices other):
        for k in other.data:
            self.data.erase(k)
        return self

    def __sub__(indices self, indices other):
        cdef indices result = type(self)()
        for k in self.data:
            if not other.data.count(k):
                result.data.insert(k)
        return result

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

    Provides a memory efficient Counter interface, with optimized conversion between numpy arrays.

    :param keys: optional iterable of keys
    :param values: optional scalar or iterable of values
    """
    cdef unordered_map[Py_ssize_t, double] data

    def __init__(self, keys=(), values=1.0):
        self.update(keys, values)

    def __repr__(self):
        return 'vector({}, {})'.format(self.keys(), self.values())

    def __len__(self):
        return self.data.size()

    def __getitem__(self, Py_ssize_t key):
        return self.data[key]

    def __setitem__(self, Py_ssize_t key, value):
        self.data[key] = value

    def __delitem__(self, Py_ssize_t key):
        self.data.erase(key)

    def __contains__(self, Py_ssize_t key):
        return self.data.count(key)

    def __iter__(self):
        for p in self.data:
            yield p.first

    cdef issubset(self, vector other):
        for p in self.data:
            if p.second != other.data[p.first]:
                return False
        return True

    def __eq__(self, other):
        return isinstance(other, vector) and len(self) == len(other) and self.issubset(other)

    def __cmp__(self, other):
        raise TypeError("ordered comparison unsupported")

    def items(self):
        """Return zipped keys and values."""
        return zip(self.keys(), self.values())

    def clear(self):
        """Remove all items."""
        self.data.clear()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def keys(self):
        """Return keys as numpy array."""
        result = np.empty(len(self), dtype)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        for p in self.data:
            arr[i] = p.first
            i += 1
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def values(self, dtype=float):
        """Return values as numpy array."""
        result = np.empty(len(self), float)
        cdef double [:] arr = result
        cdef Py_ssize_t i = 0
        for p in self.data:
            arr[i] = p.second
            i += 1
        return result if dtype is float else result.astype(dtype)
    __array__ = values

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef fromarrays(self, Py_ssize_t [:] keys, double [:] values):
        cdef Py_ssize_t i
        for i in range(min(keys.size, values.size)):
            self.data[keys[i]] += values[i]

    def update(self, keys, values=1.0):
        """Update from vector, arrays, mapping, or keys with scalar."""
        if isinstance(keys, vector):
            self += keys
        elif isinstance(keys, np.ndarray):
            if not isinstance(values, np.ndarray):
                values = np.full(len(keys), values)
            self.fromarrays(keys, values.astype(float))
        elif isinstance(keys, collections.Mapping):
            for key in keys:
                self.data[key] += keys[key]
        else:
            for key in keys:
                self.data[key] += values

    def __neg__(self):
        cdef vector result = type(self)()
        for p in self.data:
            result.data[p.first] = -p.second
        return result

    def __abs__(self):
        cdef vector result = type(self)()
        for p in self.data:
            result.data[p.first] = abs(p.second)
        return result

    def remove(self, double value=0):
        """Remove all matching values."""
        cdef int count = 0
        for p in self.data:
            if p.second == value:
                count += self.data.erase(p.first)
        return count

    cdef iadd(self, double value):
        for p in self.data:
            self.data[p.first] = p.second + value
        return self

    def __iadd__(self, value):
        if not isinstance(value, vector):
            return self.iadd(value)
        cdef vector other = value
        for p in other.data:
            self.data[p.first] += p.second
        return self

    def __add__(self, value):
        return type(self)(self).__iadd__(value)

    def __isub__(self, value):
        if not isinstance(value, vector):
            return self.iadd(-value)
        cdef vector other = value
        for p in other.data:
            self.data[p.first] -= p.second
        return self

    def __sub__(self, value):
        return type(self)(self).__isub__(value)

    cdef imul(self, double value):
        for p in self.data:
            self.data[p.first] = p.second * value
        return self

    def __imul__(self, value):
        if not isinstance(value, vector):
            return self.imul(value)
        cdef vector other = value
        for p in self.data:
            if other.data.count(p.first):
                self.data[p.first] = p.second * other.data[p.first]
            else:
                self.data.erase(p.first)
        return self

    def __mul__(vector self, value):
        if not isinstance(value, vector):
            return type(self)(self).__imul__(value)
        cdef vector other = value
        if len(other) < len(self):
            return other * self
        cdef vector result = type(self)()
        for p in self.data:
            if other.data.count(p.first):
                result.data[p.first] = p.second * other.data[p.first]
        return result

    def __itruediv__(self, double value):
        for p in self.data:
            self.data[p.first] = p.second / value
        return self

    def __truediv__(self, value):
        return type(self)(self).__itruediv__(value)

    def __ipow__(self, double value):
        for p in self.data:
            self.data[p.first] = p.second ** value
        return self

    def __pow__(self, value, modulo):
        if modulo is not None:
            raise TypeError("pow() with modulo unsupported")
        return type(self)(self).__ipow__(value)

    @classmethod
    def fromdense(cls, values):
        """Return vector from a dense array representation."""
        keys, = np.nonzero(values)
        return cls(keys, values[keys])

    def todense(self, size=None, dtype=float):
        """Return a dense array representation of vector."""
        keys = self.keys()
        result = np.zeros(keys.max() + 1 if size is None else size, dtype)
        result[keys] = self.values()
        return result
