from __future__ import division
import numpy as np
import pytest
from spector import vector


def test_basic():
    assert str(vector()) == 'vector([], [])'
    vec = vector(range(3))
    assert len(vec) == 3
    assert np.array_equal(vec.keys(), np.array([2, 1, 0]))
    assert np.array_equal(vec.values(), np.array([1, 1, 1]))
    assert np.array(vec).dtype == 'float64'
    assert np.array(vec, dtype=int).dtype == 'int64'
    assert vec[0] == 1.0
    vec[0] += 1.5
    assert vec[0] == 2.5
    assert 0 in vec
    del vec[0]
    assert 0 not in vec
    assert vec[0] == 0.0
    assert 0 in vec
    with pytest.raises(TypeError):
        vec[None]
    assert set(vec) == {0, 1, 2}
    assert dict(vec.items()) == {0: 0.0, 1: 1.0, 2: 1.0}
    vec.clear()
    assert not vec
    vec = vector(vector(range(3), 2.0))
    assert np.array_equal(vec.values(), np.array([2.0, 2.0, 2.0]))
    vec = vector({0: 0.0, 1: 1.0})
    assert set(vec.values()) == {0.0, 1.0, 0.0}
    vec.update(vector([1, 2]).keys())
    assert dict(vec.items()) == {0: 0.0, 1: 2.0, 2: 1.0}


def test_cmp():
    vec = vector(range(3))
    assert vec == vec == vector(range(3))
    assert vec != vector(range(3), 2)
    assert vec != vector(range(2))
    assert vec != vector(range(4))
    with pytest.raises(TypeError):
        vec <= vec


def test_dense():
    arr = np.array(range(4))
    vec = vector.fromdense(arr)
    assert dict(vec) == {1: 1, 2: 2, 3: 3}
    assert np.array_equal(vec.todense(), arr)
    assert list(vec.todense(5)) == [0, 1, 2, 3, 0]
    with pytest.raises(IndexError):
        vec.todense(3)


def test_math():
    vec = vector(range(3), 1.0)
    vec += 1
    assert vec == vector(range(3), 2.0)
    vec -= 1
    assert vec == vector(range(3), 1.0)
    vec *= 2
    assert vec == vector(range(3), 2.0)
    vec **= 3
    assert vec == vector(range(3), 8.0)
    vec /= 2
    assert vec == vector(range(3), 4.0)

    vec += vector([3], 4.0)
    assert vec == vector(range(4), 4.0)
    vec *= vector([3, 4], 2.0)
    assert vec == vector([3], 8.0)
    vec -= vec
    assert vec == vector([3], 0.0)

    vec = vector(range(3), 1.0)
    assert vec + 1 == vector(range(3), 2.0)
    assert vec - 1 == vector(range(3), 0.0)
    assert vec * 2 == vector(range(3), 2.0)
    assert (vec + 1) ** 3 == vector(range(3), 8.0)
    assert (vec + 1) / 2 == vector(range(3), 1.0)
    with pytest.raises(TypeError):
        pow(vec, 2, 2)

    assert vec + vector([2, 3], 2.0) == vector({0: 1.0, 1: 1.0, 2: 3.0, 3: 2.0})
    assert vec - vector([2, 3], 1.0) == vector({0: 1.0, 1: 1.0, 2: 0.0, 3: -1.0})
    other = vector([2, 3], 2.0)
    assert vec * other == other * vec == vector({2: 2.0})
    assert vec.__matmul__(other) == other.dot(vec) == 2.0


def test_unary():
    vec = vector({0: -1, 1: 0, 2: 1})
    assert -vec == vector({0: 1, 1: 0, 2: -1})
    assert abs(vec) == vector({0: 1, 1: 0, 2: 1})
    assert vec.remove() == 1
    assert vec == vector({0: -1, 2: 1})
    assert vec.remove(1) == 1
    assert vec == vector({0: -1})
    assert vec.remove() == 0
    assert vec == vector({0: -1})
