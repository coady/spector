# distutils: language=c++
import collections
import numpy as np
from cython.operator cimport dereference as deref
from libc.math cimport fmin, fmax, pow
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
cimport cython
try:
    from future_builtins import zip
except ImportError:
    pass


cdef inline double fadd(double x, double y) nogil:
    return x + y


cdef inline double fmul(double x, double y) nogil:
    return x * y


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

    cdef bool issubset(self, indices other) nogil:
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

    cdef bool intersects(self, indices other) nogil:
        for k in self.data:
            if other.data.count(k):
                return True
        return False

    def isdisjoint(self, indices other):
        """Return whether two indices have a null intersection."""
        return not (other.intersects(self) if len(other) < len(self) else self.intersects(other))

    def add(self, Py_ssize_t key):
        """Add an index key."""
        return self.data.insert(key).second

    def discard(self, Py_ssize_t key):
        """Remove an index key, if present."""
        return self.data.erase(key)

    def clear(self):
        """Remove all indices."""
        with nogil:
            self.data.clear()

    @cython.wraparound(False)
    def __array__(self):
        """Return keys as numpy array."""
        result = np.empty(len(self), np.intp)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for k in self.data:
                arr[i] = k
                i += 1
        return result[:i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef fromarray(self, Py_ssize_t [:] keys):
        cdef Py_ssize_t n = keys.size
        with nogil:
            for i in range(n):
                self.data.insert(keys[i])

    def update(self, keys):
        """Update from indices, array, or iterable."""
        if isinstance(keys, indices):
            self |= keys
        elif isinstance(keys, np.ndarray):
            self.fromarray(keys.astype(np.intp, casting='safe', copy=False))
        else:
            for key in keys:
                self.data.insert(key)

    def __ior__(self, indices other):
        with nogil:
            for k in other.data:
                self.data.insert(k)
        return self

    def __or__(self, indices other):
        return type(self)(self).__ior__(other)

    def __ixor__(self, indices other):
        with nogil:
            for k in other.data:
                if not self.data.insert(k).second:
                    self.data.erase(k)
        return self

    def __xor__(self, indices other):
        return type(self)(self).__ixor__(other)

    cdef void ifilter(self, indices other, int count) nogil:
        for k in self.data:
            if other.data.count(k) != count:
                self.data.erase(k)

    def __iand__(self, other):
        self.ifilter(other, 1)
        return self

    cdef filter(self, indices other, int count):
        cdef indices result = type(self)()
        with nogil:
            for k in self.data:
                if other.data.count(k) == count:
                    result.data.insert(k)
        return result

    def __and__(indices self, indices other):
        return other.filter(self, 1) if len(other) < len(self) else self.filter(other, 1)

    def __isub__(self, indices other):
        if len(other) < len(self):
            with nogil:
                for k in other.data:
                    self.data.erase(k)
        else:
            self.ifilter(other, 0)
        return self

    def __sub__(indices self, other):
        return self.filter(other, 0)

    @classmethod
    def fromdense(cls, values):
        """Return indices from a dense array representation."""
        return cls(*np.nonzero(values))

    def todense(self, size=None):
        """Return a dense array representation of indices."""
        keys = np.array(self)
        result = np.zeros(keys.max() + 1 if size is None else size, np.bool)
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

    def __getitem__(self, key):
        if isinstance(key, collections.Iterable):
            return type(self)(key).__imul__(self)  # keys are typically a subset
        return self.get(key)

    def __setitem__(self, key, value):
        if isinstance(key, collections.Iterable):
            other = vector(key, value)
            with nogil:
                for p in other.data:
                    self.data[p.first] = p.second
        else:
            self.data[key] = value

    def __delitem__(self, key):
        keys = indices(key if isinstance(key, collections.Iterable) else [key])
        with nogil:
            for k in keys.data:
                self.data.erase(k)

    def __contains__(self, Py_ssize_t key):
        return self.data.count(key)

    def __iter__(self):
        for p in self.data:
            yield p.first

    cdef double get(self, Py_ssize_t key) nogil:
        it = self.data.find(key)
        return deref(it).second if it != self.data.end() else 0.0

    @cython.wraparound(False)
    cdef apply(self, vector other):
        result = np.empty(len(self), float)
        cdef double [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for p in self.data:
                arr[i] = other.get(p.first)
                i += 1
        return result[:i]

    def map(self, ufunc, *args, **kwargs):
        """Return element-wise array of values from applying function across vectors."""
        args = [self.apply(arg) if isinstance(arg, vector) else arg for arg in args]
        return ufunc(self, *args, **kwargs)

    def filter(self, ufunc, *args, **kwargs):
        """Return element-wise array of keys from applying predicate across vectors."""
        return self.keys()[self.map(ufunc, *args, **kwargs)]

    def equal(self, vector other):
        """Return whether vectors are equal as scalar bool; == is element-wise."""
        return self.data == other.data

    def __eq__(self, value):
        return self.filter(np.equal, value)

    def __ne__(self, value):
        return self.filter(np.not_equal, value)

    def __lt__(self, value):
        return self.filter(np.less, value)

    def __le__(self, value):
        return self.filter(np.less_equal, value)

    def __gt__(self, value):
        return self.filter(np.greater, value)

    def __ge__(self, value):
        return self.filter(np.greater_equal, value)

    def items(self):
        """Return zipped keys and values."""
        return zip(self.keys(), self.values())

    def clear(self):
        """Remove all items."""
        with nogil:
            self.data.clear()

    @cython.wraparound(False)
    def keys(self):
        """Return keys as numpy array."""
        result = np.empty(len(self), np.intp)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for p in self.data:
                arr[i] = p.first
                i += 1
        return result[:i]

    @cython.wraparound(False)
    def values(self, dtype=float):
        """Return values as numpy array."""
        result = np.empty(len(self), float)
        cdef double [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for p in self.data:
                arr[i] = p.second
                i += 1
        return result.astype(dtype, copy=False)[:i]
    __array__ = values

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef fromarrays(self, Py_ssize_t [:] keys, double [:] values):
        cdef Py_ssize_t n = min(keys.size, values.size)
        with nogil:
            for i in range(n):
                self.data[keys[i]] += values[i]

    def update(self, keys, values=1.0):
        """Update from vector, arrays, mapping, or keys with scalar."""
        if isinstance(keys, vector):
            self += keys
        elif isinstance(keys, np.ndarray):
            values = np.asfarray(values if isinstance(values, np.ndarray) else np.full(len(keys), values))
            self.fromarrays(keys.astype(np.intp, casting='safe', copy=False), values)
        elif isinstance(keys, collections.Mapping):
            for key in keys:
                self.data[key] += keys[key]
        else:
            for key in keys:
                self.data[key] += values

    def __neg__(self):
        return type(self)(self.keys(), np.negative(self))

    def __abs__(self):
        return type(self)(self.keys(), np.absolute(self))

    def minimum(self, value):
        """Return element-wise minimum vector."""
        return type(self)(self.keys(), self.map(np.minimum, value))

    def maximum(self, value):
        """Return element-wise maximum vector."""
        return type(self)(self.keys(), self.map(np.maximum, value))

    cdef void imap(self, double value, double (*op)(double, double) nogil) nogil:
        for p in self.data:
            self.data[p.first] = op(p.second, value)

    cdef void ior(self, vector other, double (*op)(double, double) nogil) nogil:
        for p in other.data:
            self.data[p.first] = op(self.data[p.first], p.second)

    def __iadd__(self, value):
        if isinstance(value, vector):
            self.ior(value, fadd)
        else:
            self.imap(value, fadd)
        return self

    def __add__(self, value):
        return type(self)(self).__iadd__(value)

    def __isub__(self, double value):
        return self.__iadd__(-value)

    def __sub__(self, value):
        return type(self)(self).__isub__(value)

    cdef void iand(self, vector other, double (*op)(double, double) nogil) nogil:
        for p in self.data:
            it = other.data.find(p.first)
            if it != other.data.end():
                self.data[p.first] = op(p.second, deref(it).second)
            else:
                self.data.erase(p.first)

    def __imul__(self, value):
        if isinstance(value, vector):
            self.iand(value, fmul)
        else:
            self.imap(value, fmul)
        return self

    cdef and_(self, vector other, double (*op)(double, double) nogil):
        cdef vector result = type(self)()
        with nogil:
            for p in self.data:
                it = other.data.find(p.first)
                if it != other.data.end():
                    result.data[p.first] = op(p.second, deref(it).second)
        return result

    def __mul__(vector self, value):
        if not isinstance(value, vector):
            return type(self)(self).__imul__(value)
        return (value * self) if len(value) < len(self) else self.and_(value, fmul)

    def __ior__(self, other):
        self.ior(other, fmax)
        return self

    def __or__(self, vector other):
        return type(self)(self).__ior__(other)

    def __iand__(self, other):
        self.iand(other, fmin)
        return self

    def __and__(vector self, other):
        return (other & self) if len(other) < len(self) else self.and_(other, fmin)

    def __ixor__(self, vector other):
        with nogil:
            for p in other.data:
                if not self.data.erase(p.first):
                    self.data[p.first] = p.second
        return self

    def __xor__(self, vector other):
        return type(self)(self).__ixor__(other)

    def difference(self, keys):
        """Provisional set difference; return vector without keys."""
        ind = indices(keys)
        cdef vector result = type(self)()
        with nogil:
            for p in self.data:
                if not ind.data.count(p.first):
                    result.data[p.first] = p.second
        return result

    def dot(self, vector other):
        """Return dot product."""
        if len(other) < len(self):
            return other.dot(self)
        cdef double total = 0.0
        with nogil:
            for p in self.data:
                total += p.second * other.get(p.first)
        return total
    __matmul__ = dot

    def __itruediv__(self, double value):
        return self.__imul__(1.0 / value)

    def __truediv__(self, value):
        return type(self)(self).__itruediv__(value)

    def __ipow__(self, value):
        self.imap(value, pow)
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
