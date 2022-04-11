# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False
import operator
from typing import Mapping
import numpy as np
import cython
from cython.operator import dereference, postincrement
from libc.math cimport fmin, fmax, pow
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set


cdef inline double fadd(double x, double y) nogil:
    return x + y


cdef inline double fmul(double x, double y) nogil:
    return x * y


cdef inline bool flt(double x, double y) nogil:
    return x < y


cdef inline bool fgt(double x, double y) nogil:
    return x > y


def asiarray(keys):
    return np.asarray(keys).astype(np.intp, casting='safe', copy=False)


cdef indices asindices(keys):
    return keys if isinstance(keys, indices) else indices(keys, len(keys))


def arggroupby(const Py_ssize_t[:] keys):
    """Generate unique keys with corresponding index arrays."""
    values: Py_ssize_t[:] = np.empty(keys.size, np.intp)
    size: Py_ssize_t = 0
    sizes: unordered_map[Py_ssize_t, Py_ssize_t]
    with nogil:
        for i in range(keys.shape[0]):
            postincrement(sizes[keys[i]])
        for p in sizes:
            sizes[p.first] = size
            size += p.second
        for i in range(keys.shape[0]):
            values[postincrement(sizes[keys[i]])] = i
    start = 0
    for p in sizes:
        yield p.first, values[start:p.second]
        start = p.second


cdef class indices:
    """A sparse boolean array, i.e., set of indices.

    Provides a memory efficient set interface, with optimized conversion between numpy arrays.

    Args:
        keys (Iterable): optional iterable of keys
    """
    data: unordered_set[Py_ssize_t]

    def __init__(self, keys=(), length_hint=0):
        self.resize(length_hint)
        self.update(keys)

    def __repr__(self):
        return f'indices({np.array(self)})'

    def __len__(self):
        return self.data.size()

    def __contains__(self, key: Py_ssize_t):
        return self.data.count(key)

    def __iter__(self):
        for k in self.data:
            yield k

    @cython.nogil
    @cython.cfunc
    def all(self, other: indices, count: size_t) -> bool:
        with nogil:
            for k in self.data:
                if other.data.count(k) != count:
                    return False
        return True

    def __eq__(self, other):
        return isinstance(other, indices) and len(self) == len(other) and self.all(other, 1)

    def __le__(self, other: indices):
        return len(self) <= len(other) and self.all(other, 1)

    def __lt__(self, other: indices):
        return len(self) < len(other) and self.all(other, 1)

    def isdisjoint(self, other: indices):
        """Return whether two indices have a null intersection."""
        self, other = sorted([self, other], key=len)
        return self.all(other, 0)

    def add(self, key: Py_ssize_t):
        """Add an index key."""
        return self.data.insert(key).second

    def discard(self, key: Py_ssize_t):
        """Remove an index key, if present."""
        return self.data.erase(key)

    def clear(self):
        """Remove all indices."""
        with nogil:
            self.data.clear()

    @cython.boundscheck(True)
    def __array__(self, dtype=np.intp):
        """Return keys as numpy array."""
        result = np.empty(len(self), np.intp)
        arr: Py_ssize_t[:] = result
        i: Py_ssize_t = 0
        with nogil:
            for k in self.data:
                arr[postincrement(i)] = k
        return result[:i].astype(dtype, copy=False)

    @cython.nogil
    @cython.cfunc
    def resize(self, count: size_t) -> void:
        if count >= (self.data.bucket_count() * 2):
            self.data.reserve(count)

    cdef void fromarray(self, const Py_ssize_t[:] keys, size_t length_hint=0) nogil:
        with nogil:
            self.resize(length_hint)
            for i in range(keys.shape[0]):
                self.data.insert(keys[i])

    def update(self, *others):
        """Update from indices, arrays, or iterables."""
        for keys in others:
            if isinstance(keys, indices):
                self |= keys
            elif isinstance(keys, vector):
                self.fromarray(keys.keys(), len(keys))
            elif hasattr(keys, '__array__'):
                self.fromarray(asiarray(keys))
            else:
                for key in keys:
                    self.data.insert(key)

    def union(self, *others):
        """Return the union of sets as a new set."""
        self = type(self)(self)
        self.update(*others)
        return self

    cdef select(self, const Py_ssize_t[:] keys, size_t count):
        result = np.empty(keys.size, np.intp)
        arr: Py_ssize_t[:]  = result
        i: Py_ssize_t = 0
        with nogil:
            for j in range(keys.shape[0]):
                if self.data.count(keys[j]) == count:
                    arr[postincrement(i)] = keys[j]
        return result[:i]

    def intersection(self, *others):
        """Return the intersection of sets as a new set."""
        result = self
        for other in sorted(others, key=operator.length_hint):
            if isinstance(other, indices) and len(other) >= len(result):
                result = (<indices> other).select(asiarray(result), 1)
            else:
                result = asindices(result).select(asiarray(other), 1)
        return indices(result, len(result))

    def difference(self, *others):
        """Return the difference of sets as a new set."""
        result = np.asarray(self)
        for other in sorted(others, key=operator.length_hint, reverse=True):
            result = asindices(other).select(result, 0)
        return asindices(result)

    def __ior__(self, other: indices):
        with nogil:
            self.resize(other.data.size())
            for k in other.data:
                self.data.insert(k)
        return self

    def __or__(self, other: indices):
        return type(self)(self).__ior__(other)

    def __ixor__(self, other: indices):
        with nogil:
            for k in other.data:
                if not self.data.insert(k).second:
                    self.data.erase(k)
        return self

    def __xor__(self, other: indices):
        return type(self)(self).__ixor__(other)

    @cython.nogil
    @cython.cfunc
    def ifilter(self, other: indices, count: size_t) -> void:
        with nogil:
            for k in self.data:
                if other.data.count(k) != count:
                    self.data.erase(k)

    def __iand__(self, other: indices):
        self.ifilter(other, 1)
        return self

    @cython.cfunc
    def filter(self, other: indices, count: size_t):
        result: indices = type(self)(length_hint=not count and max(0, len(self) - len(other)))
        with nogil:
            for k in self.data:
                if other.data.count(k) == count:
                    result.data.insert(k)
        return result

    def __and__(self: indices, other: indices):
        self, other = sorted([self, other], key=len)
        return self.filter(other, 1)

    def __isub__(self, indices other):
        if len(other) < len(self):
            with nogil:
                for k in other.data:
                    self.data.erase(k)
        else:
            self.ifilter(other, 0)
        return self

    def __sub__(self: indices, other: indices):
        return self.filter(other, 0)

    def __matmul__(self: indices, other: indices):
        """Return binary dot product, i.e., intersection count."""
        self, other = sorted([self, other], key=len)
        total: size_t = 0
        with nogil:
            for k in self.data:
                total += <size_t> other.data.count(k)
        return total

    @cython.boundscheck(True)
    def dot(self, *others):
        """Return the intersection count of sets."""
        others = sorted(others, key=operator.length_hint)
        if not others:
            return len(self)
        if not isinstance(others[-1], indices):
            return len(self.intersection(*others))
        if len(others) > 1:
            self = self.intersection(*others[:-1])
        return self @ others[-1]

    @classmethod
    def fromdense(cls, values):
        """Return indices from a dense array representation."""
        keys, = np.nonzero(values)
        return cls(keys, len(keys))

    def todense(self, minlength=0, dtype=np.bool_):
        """Return a dense array representation of indices."""
        return np.bincount(self, minlength=minlength).astype(dtype, copy=False)


cdef class vector:
    """A sparse array of index keys mapped to numeric values.

    Provides a memory efficient Counter interface, with optimized conversion between numpy arrays.

    Args:
        keys (Iterable[int]): optional iterable of keys
        values: optional scalar or iterable of values
    """
    data: unordered_map[Py_ssize_t, double]

    def __init__(self, keys=(), values=1.0, length_hint=0):
        self.resize(length_hint)
        self.update(keys, values)

    def __repr__(self):
        return f'vector({self.keys()}, {self.values()})'

    def __len__(self):
        return self.data.size()

    def __getitem__(self, key):
        try:
            return self.get(key)
        except TypeError:
            result = type(self)(np.asarray(key), 0.0)
        (<vector> result).iand(self, fadd)
        return result

    def __setitem__(self, key, value: double):
        arr: Py_ssize_t[:] = np.repeat(asiarray(key), 1)
        with nogil:
            for i in range(arr.shape[0]):
                self.data[arr[i]] = value

    def __delitem__(self, key):
        arr: Py_ssize_t[:] = np.repeat(asiarray(key), 1)
        with nogil:
            for i in range(arr.shape[0]):
                self.data.erase(arr[i])

    def __contains__(self, key: Py_ssize_t):
        return self.data.count(key)

    def __iter__(self):
        for p in self.data:
            yield p.first

    @cython.nogil
    @cython.cfunc
    def get(self, key: Py_ssize_t) -> double:
        it = self.data.find(key)
        return dereference(it).second if it != self.data.end() else 0.0

    @cython.boundscheck(True)
    @cython.cfunc
    def apply(self, other: vector):
        result = np.empty(len(self), float)
        arr: double[:] = result
        i: Py_ssize_t = 0
        with nogil:
            for p in self.data:
                arr[postincrement(i)] = other.get(p.first)
        return result[:i]

    def map(self, ufunc, *args, **kwargs):
        """Return element-wise array of values from applying function across vectors."""
        args = [self.apply(arg) if isinstance(arg, vector) else arg for arg in args]
        return ufunc(self.values(), *args, **kwargs)

    def filter(self, ufunc, *args, **kwargs):
        """Return element-wise array of keys from applying predicate across vectors."""
        return self.keys()[self.map(ufunc, *args, **kwargs)]

    def equal(self, other: vector):
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

    @cython.boundscheck(True)
    def keys(self):
        """Return keys as numpy array."""
        result = np.empty(len(self), np.intp)
        arr: Py_ssize_t[:] = result
        i: Py_ssize_t = 0
        with nogil:
            for p in self.data:
                arr[postincrement(i)] = p.first
        return result[:i]

    @cython.boundscheck(True)
    def values(self, dtype=float):
        """Return values as numpy array."""
        result = np.empty(len(self), float)
        arr: double[:] = result
        i: Py_ssize_t = 0
        with nogil:
            for p in self.data:
                arr[postincrement(i)] = p.second
        return result[:i].astype(dtype, copy=False)
    __array__ = values

    @cython.nogil
    @cython.cfunc
    def resize(self, count: size_t) -> void:
        if count >= (self.data.bucket_count() * 2):
            self.data.reserve(count)

    cdef void fromarrays(self, const Py_ssize_t[:] keys, const double[:] values, size_t length_hint=0) nogil:
        with nogil:
            self.resize(length_hint)
            for i in range(min(keys.shape[0], values.shape[0])):
                self.data[keys[i]] += values[i]

    def update(self, keys, values=1.0):
        """Update from vector, arrays, mapping, or keys with scalar."""
        if isinstance(keys, vector):
            self += keys
        elif hasattr(keys, '__array__'):
            values = np.asfarray(values)
            values = values.repeat(values.ndim or len(keys))
            self.fromarrays(asiarray(keys), values, isinstance(keys, indices) and len(keys))
        elif isinstance(keys, Mapping):
            for key in keys:
                self.data[key] += keys[key]
        else:
            for key in keys:
                self.data[key] += values

    @cython.cfunc
    def replace(self, values):
        return type(self)(self.keys(), values, len(self))

    def __neg__(self):
        return self.replace(np.negative(self))

    def __abs__(self):
        return self.replace(np.absolute(self))

    def minimum(self, value):
        """Return element-wise minimum vector."""
        return self.replace(self.map(np.minimum, value))

    def maximum(self, value):
        """Return element-wise maximum vector."""
        return self.replace(self.map(np.maximum, value))

    cdef void imap(self, double value, double (*op)(double, double) nogil) nogil:
        with nogil:
            for p in self.data:
                self.data[p.first] = op(p.second, value)

    cdef void ior(self, vector other, double (*op)(double, double) nogil) nogil:
        with nogil:
            self.resize(other.data.size())
            for p in other.data:
                self.data[p.first] = op(self.data[p.first], p.second)

    def __iadd__(self, value):
        if isinstance(value, vector):
            self.ior(value, fadd)
        else:
            self.imap(value, fadd)
        return self

    @cython.cfunc
    def rop(self, ufunc, value: double):
        return self.replace(ufunc(value, self))

    def __add__(self, value):
        if not isinstance(self, vector):
            return (<vector> value).rop(np.add, self)
        return type(self)(self).__iadd__(value)

    def __isub__(self, value: double):
        return self.__iadd__(-value)

    def __sub__(self, value):
        if not isinstance(self, vector):
            return (<vector> value).rop(np.subtract, self)
        return type(self)(self).__isub__(value)

    cdef void iand(self, vector other, double (*op)(double, double) nogil) nogil:
        with nogil:
            for p in self.data:
                it = other.data.find(p.first)
                if it != other.data.end():
                    self.data[p.first] = op(p.second, dereference(it).second)
                else:
                    self.data.erase(p.first)

    def __imul__(self, value):
        if isinstance(value, vector):
            self.iand(value, fmul)
        else:
            self.imap(value, fmul)
        return self

    cdef and_(self, vector other, double (*op)(double, double) nogil):
        result: vector = type(self)()
        with nogil:
            for p in self.data:
                it = other.data.find(p.first)
                if it != other.data.end():
                    result.data[p.first] = op(p.second, dereference(it).second)
        return result

    def __mul__(self, value):
        if not isinstance(self, vector):
            return (<vector> value).rop(np.multiply, self)
        if not isinstance(value, vector):
            return type(self)(self).__imul__(value)
        self, other = sorted([self, value], key=len)
        return (<vector> self).and_(other, fmul)

    def __ior__(self, other: vector):
        self.ior(other, fmax)
        return self

    def __or__(self, other: vector):
        return type(self)(self).__ior__(other)

    def __iand__(self, other: vector):
        self.iand(other, fmin)
        return self

    def __and__(self: vector, other: vector):
        self, other = sorted([self, other], key=len)
        return self.and_(other, fmin)

    def __ixor__(self, other: vector):
        with nogil:
            for p in other.data:
                if not self.data.erase(p.first):
                    self.data[p.first] = p.second
        return self

    def __xor__(self, other: vector):
        return type(self)(self).__ixor__(other)

    def difference(self, *others):
        """Provisional set difference; return vector without keys."""
        other: indices = indices().union(*others)
        result: vector = type(self)(length_hint=max(0, len(self) - len(other)))
        with nogil:
            for p in self.data:
                if not other.data.count(p.first):
                    result.data[p.first] = p.second
        return result

    def __matmul__(self: vector, other: vector):
        """Return vector dot product."""
        self, other = sorted([self, other], key=len)
        total: double = 0.0
        with nogil:
            for p in self.data:
                total += p.second * other.get(p.first)
        return total

    def __itruediv__(self, value: double):
        return self.__imul__(1.0 / value)

    def __truediv__(self, value):
        if not isinstance(self, vector):
            return (<vector> value).rop(np.true_divide, self)
        return type(self)(self).__itruediv__(value)

    def __ipow__(self, value: double):
        self.imap(value, pow)
        return self

    def __pow__(self, value, modulo):
        if modulo is not None:
            raise TypeError("pow() with modulo unsupported")
        if not isinstance(self, vector):
            return (<vector> value).rop(np.power, self)
        return type(self)(self).__ipow__(value)

    @classmethod
    def fromdense(cls, values):
        """Return vector from a dense array representation."""
        values = np.asfarray(values)
        keys, = np.nonzero(values)
        return cls(keys, values[keys], len(keys))

    def todense(self, minlength=0, dtype=float):
        """Return a dense array representation of vector."""
        return np.bincount(self.keys(), self, minlength).astype(dtype, copy=False)

    cdef double reduce(self, double (*op)(double, double) nogil, double initial) nogil:
        with nogil:
            for p in self.data:
                initial = op(initial, p.second)
        return initial

    def sum(self, initial=0.0, dtype=float, **kwargs):
        """Return sum of values."""
        return dtype(self.reduce(fadd, initial))

    cdef argcmp(self, bool (*cmp)(double, double) nogil):
        empty: bool = True
        with nogil:
            for p in self.data:
                if empty or cmp(p.second, value):
                    empty, key, value = False, p.first, p.second
        if empty:
            raise ValueError("min/max of an empty vector")
        return key, value

    @cython.boundscheck(True)
    def min(self, initial=None, **kwargs):
        """Return minimum value."""
        return self.argcmp(flt)[1] if initial is None else self.reduce(fmin, initial)

    @cython.boundscheck(True)
    def max(self, initial=None, **kwargs):
        """Return maximum value."""
        return self.argcmp(fgt)[1] if initial is None else self.reduce(fmax, initial)

    @cython.boundscheck(True)
    def argmin(self, **kwargs):
        """Return key with minimum value."""
        return self.argcmp(flt)[0]

    @cython.boundscheck(True)
    def argmax(self, **kwargs):
        """Return key with maximum value."""
        return self.argcmp(fgt)[0]

    def argpartition(self, kth, **kwargs):
        """Return keys partitioned by values."""
        return self.filter(np.argpartition, kth, **kwargs)

    def argsort(self, **kwargs):
        """Return keys sorted by values."""
        return self.filter(np.argsort, **kwargs)

    def nonzero(self):
        return self.filter(np.nonzero),
