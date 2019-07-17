# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False
import collections
import numpy as np
from cython.operator cimport dereference as deref, postincrement as inc
from libc.math cimport fmin, fmax, pow
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
cimport cython


def length_hint(iterable):
    try:
        return len(iterable)
    except TypeError:
        return 0


try:
    from future_builtins import zip
except ImportError:
    assert length_hint(None) == 0  # coverage
    from operator import length_hint


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


def arggroupby(Py_ssize_t [:] keys):
    """Generate unique keys with corresponding index arrays."""
    grouped = np.empty(keys.size, np.intp)
    cdef Py_ssize_t [:] values = grouped
    cdef Py_ssize_t size = 0
    cdef unordered_map[Py_ssize_t, Py_ssize_t] sizes
    with nogil:
        for i in range(keys.shape[0]):
            inc(sizes[keys[i]])
        for p in sizes:
            sizes[p.first] = size
            size += p.second
        for i in range(keys.shape[0]):
            values[inc(sizes[keys[i]])] = i
    start = 0
    for p in sizes:
        yield p.first, grouped[start:p.second]
        start = p.second


cdef class indices:
    """A sparse boolean array, i.e., set of indices.

    Provides a memory efficient set interface, with optimized conversion between numpy arrays.

    :param keys: optional iterable of keys
    """
    cdef unordered_set[Py_ssize_t] data

    def __init__(self, keys=(), length_hint=0):
        self.resize(length_hint)
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

    cdef bool all(self, indices other, size_t count) nogil:
        with nogil:
            for k in self.data:
                if other.data.count(k) != count:
                    return False
        return True

    def __eq__(self, other):
        return isinstance(other, indices) and len(self) == len(other) and self.all(other, 1)

    def __le__(self, indices other):
        return len(self) <= len(other) and self.all(other, 1)

    def __lt__(self, indices other):
        return len(self) < len(other) and self.all(other, 1)

    def isdisjoint(self, indices other):
        """Return whether two indices have a null intersection."""
        self, other = sorted([self, other], key=len)
        return self.all(other, 0)

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

    @cython.boundscheck(True)
    def __array__(self, dtype=np.intp):
        """Return keys as numpy array."""
        result = np.empty(len(self), np.intp)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for k in self.data:
                arr[inc(i)] = k
        return result[:i].astype(dtype, copy=False)

    cdef void resize(self, size_t count) nogil:
        if count >= (self.data.bucket_count() * 2):
            self.data.reserve(count)

    cdef void fromarray(self, Py_ssize_t [:] keys, size_t length_hint=0) nogil:
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

    cdef select(self, Py_ssize_t [:] keys, size_t count):
        result = np.empty(keys.size, np.intp)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for j in range(keys.shape[0]):
                if self.data.count(keys[j]) == count:
                    arr[inc(i)] = keys[j]
        return result[:i]

    def intersection(self, *others):
        """Return the intersection of sets as a new set."""
        result = np.asarray(self)
        for other in sorted(others, key=length_hint):
            if isinstance(other, indices):
                result = (<indices> other).select(result, 1)
            else:
                result = indices(result, len(result)).select(asiarray(other), 1)
        return type(self)(result, len(result))

    def difference(self, *others):
        """Return the difference of sets as a new set."""
        result = np.asarray(self)
        for other in sorted(others, key=length_hint, reverse=True):
            result = (<indices> other if isinstance(other, indices) else indices(other)).select(result, 0)
        return type(self)(result, len(result))

    def __ior__(self, indices other):
        with nogil:
            self.resize(other.data.size())
            for k in other.data:
                self.data.insert(k)
        return self

    def __or__(indices self, indices other):
        return type(self)(self).__ior__(other)

    def __ixor__(self, indices other):
        with nogil:
            for k in other.data:
                if not self.data.insert(k).second:
                    self.data.erase(k)
        return self

    def __xor__(indices self, indices other):
        return type(self)(self).__ixor__(other)

    cdef void ifilter(self, indices other, size_t count) nogil:
        with nogil:
            for k in self.data:
                if other.data.count(k) != count:
                    self.data.erase(k)

    def __iand__(self, indices other):
        self.ifilter(other, 1)
        return self

    cdef filter(self, indices other, size_t count):
        cdef indices result = type(self)(length_hint=not count and max(0, len(self) - len(other)))
        with nogil:
            for k in self.data:
                if other.data.count(k) == count:
                    result.data.insert(k)
        return result

    def __and__(indices self, indices other):
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

    def __sub__(indices self, indices other):
        return self.filter(other, 0)

    def dot(self, indices other):
        """For Python 2 only; @ preferred."""
        self, other = sorted([self, other], key=len)
        cdef size_t total = 0
        with nogil:
            for k in self.data:
                total += <size_t> other.data.count(k)
        return total

    def __matmul__(self, other):
        """Return binary dot product, i.e., intersection count."""
        return self.dot(other)

    @classmethod
    def fromdense(cls, values):
        """Return indices from a dense array representation."""
        keys, = np.nonzero(values)
        return cls(keys, len(keys))

    def todense(self, minlength=0, dtype=np.bool):
        """Return a dense array representation of indices."""
        return np.bincount(self, minlength=minlength).astype(dtype, copy=False)


cdef class vector:
    """A sparse array of index keys mapped to numeric values.

    Provides a memory efficient Counter interface, with optimized conversion between numpy arrays.

    :param keys: optional iterable of keys
    :param values: optional scalar or iterable of values
    """
    cdef unordered_map[Py_ssize_t, double] data

    def __init__(self, keys=(), values=1.0, length_hint=0):
        self.resize(length_hint)
        self.update(keys, values)

    def __repr__(self):
        return 'vector({}, {})'.format(self.keys(), self.values())

    def __len__(self):
        return self.data.size()

    def __getitem__(self, key):
        try:
            return self.get(key)
        except TypeError:
            result = type(self)(np.asarray(key), 0.0)
        (<vector> result).iand(self, fadd)
        return result

    def __setitem__(self, key, double value):
        cdef Py_ssize_t [:] arr = np.repeat(asiarray(key), 1)
        with nogil:
            for i in range(arr.shape[0]):
                self.data[arr[i]] = value

    def __delitem__(self, key):
        cdef Py_ssize_t [:] arr = np.repeat(asiarray(key), 1)
        with nogil:
            for i in range(arr.shape[0]):
                self.data.erase(arr[i])

    def __contains__(self, Py_ssize_t key):
        return self.data.count(key)

    def __iter__(self):
        for p in self.data:
            yield p.first

    cdef double get(self, Py_ssize_t key) nogil:
        it = self.data.find(key)
        return deref(it).second if it != self.data.end() else 0.0

    @cython.boundscheck(True)
    cdef apply(self, vector other):
        result = np.empty(len(self), float)
        cdef double [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for p in self.data:
                arr[inc(i)] = other.get(p.first)
        return result[:i]

    def map(self, ufunc, *args, **kwargs):
        """Return element-wise array of values from applying function across vectors."""
        args = [self.apply(arg) if isinstance(arg, vector) else arg for arg in args]
        return ufunc(self.values(), *args, **kwargs)

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

    @cython.boundscheck(True)
    def keys(self):
        """Return keys as numpy array."""
        result = np.empty(len(self), np.intp)
        cdef Py_ssize_t [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for p in self.data:
                arr[inc(i)] = p.first
        return result[:i]

    @cython.boundscheck(True)
    def values(self, dtype=float):
        """Return values as numpy array."""
        result = np.empty(len(self), float)
        cdef double [:] arr = result
        cdef Py_ssize_t i = 0
        with nogil:
            for p in self.data:
                arr[inc(i)] = p.second
        return result[:i].astype(dtype, copy=False)
    __array__ = values

    cdef void resize(self, size_t count) nogil:
        if count >= (self.data.bucket_count() * 2):
            self.data.reserve(count)

    cdef void fromarrays(self, Py_ssize_t [:] keys, double [:] values, size_t length_hint=0) nogil:
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
        elif isinstance(keys, collections.Mapping):
            for key in keys:
                self.data[key] += keys[key]
        else:
            for key in keys:
                self.data[key] += values

    cdef replace(self, values):
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

    cdef rop(self, ufunc, double value):
        return self.replace(ufunc(value, self))

    def __add__(self, value):
        if not isinstance(self, vector):
            return (<vector> value).rop(np.add, self)
        return type(self)(self).__iadd__(value)

    def __isub__(self, double value):
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

    def __mul__(self, value):
        if not isinstance(self, vector):
            return (<vector> value).rop(np.multiply, self)
        if not isinstance(value, vector):
            return type(self)(self).__imul__(value)
        self, other = sorted([self, value], key=len)
        return (<vector> self).and_(other, fmul)

    def __ior__(self, vector other):
        self.ior(other, fmax)
        return self

    def __or__(vector self, vector other):
        return type(self)(self).__ior__(other)

    def __iand__(self, vector other):
        self.iand(other, fmin)
        return self

    def __and__(vector self, vector other):
        self, other = sorted([self, other], key=len)
        return self.and_(other, fmin)

    def __ixor__(self, vector other):
        with nogil:
            for p in other.data:
                if not self.data.erase(p.first):
                    self.data[p.first] = p.second
        return self

    def __xor__(vector self, vector other):
        return type(self)(self).__ixor__(other)

    def difference(self, *others):
        """Provisional set difference; return vector without keys."""
        cdef indices other = indices().union(*others)
        cdef vector result = type(self)(length_hint=max(0, len(self) - len(other)))
        with nogil:
            for p in self.data:
                if not other.data.count(p.first):
                    result.data[p.first] = p.second
        return result

    def dot(self, vector other):
        """For Python 2 only; @ preferred."""
        self, other = sorted([self, other], key=len)
        cdef double total = 0.0
        with nogil:
            for p in self.data:
                total += p.second * other.get(p.first)
        return total

    def __matmul__(self, other):
        """Return vector dot product."""
        return self.dot(other)

    def __itruediv__(self, double value):
        return self.__imul__(1.0 / value)

    def __truediv__(self, value):
        if not isinstance(self, vector):
            return (<vector> value).rop(np.true_divide, self)
        return type(self)(self).__itruediv__(value)

    def __ipow__(self, double value):
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
        cdef bool empty = True
        with nogil:
            for p in self.data:
                if empty or cmp(p.second, value):
                    empty, key, value = False, p.first, p.second
        if empty:
            raise ValueError("min/max of an empty vector")
        return key, value

    def min(self, initial=None, **kwargs):
        """Return minimum value."""
        return self.argcmp(flt)[1] if initial is None else self.reduce(fmin, initial)

    def max(self, initial=None, **kwargs):
        """Return maximum value."""
        return self.argcmp(fgt)[1] if initial is None else self.reduce(fmax, initial)

    def argmin(self, **kwargs):
        """Return key with minimum value."""
        return self.argcmp(flt)[0]

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
