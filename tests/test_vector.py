from __future__ import division
import numpy as np
import pytest
from spector import indices, vector
from spector.vector import arggroupby


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
    assert 0 not in vec
    with pytest.raises(TypeError):
        vec[None]
    assert set(vec) == {1, 2}
    assert dict(vec.items()) == {1: 1.0, 2: 1.0}
    vec.clear()
    assert not vec
    vec = vector(vector(range(3), 2.0))
    assert np.array_equal(vec.values(), np.array([2.0, 2.0, 2.0]))
    vec = vector({0: 0.0, 1: 1.0})
    assert set(vec.values()) == {0.0, 1.0, 0.0}
    vec.update(vector([1, 2]).keys())
    assert dict(vec.items()) == {0: 0.0, 1: 2.0, 2: 1.0}
    assert vector(np.array([False]), np.array(['0']))
    assert vector(indices([0]))
    with pytest.raises(TypeError):
        vector(np.array([0.0]))
    with pytest.raises(ValueError):
        vector(np.array([0]), np.array([' ']))


def test_cmp():
    vec = vector(range(3))
    assert set(vec == vector(range(3))) == {0, 1, 2}
    assert np.array_equal(vec != vector(range(3)), np.empty(0))
    assert set(vec <= vector(range(2))) == {0, 1}
    assert set(vec >= vector(range(4))) == {0, 1, 2}
    assert np.array_equal(vec < vector({0: 1.0, 1: 2.0}), np.array([1]))
    assert np.array_equal(vec > vector({0: 1.0, 1: 2.0}), np.array([2]))

    vec = vector({0: 0.0, 1: 1.0, 2: 2.0})
    assert np.array_equal(vec == 1.0, np.array([1]))
    assert set(vec != 1.0) == {0, 2}
    assert set(vec <= 1.0) == {0, 1}
    assert np.array_equal(vec < 1.0, np.array([0]))
    assert set(vec >= 1.0) == {1, 2}
    assert np.array_equal(vec > 1.0, np.array([2]))

    assert vec[vec <= 1.0].equal(vector({0: 0.0, 1: 1.0}))
    assert vec[vec >= 1.0].equal(vector({1: 1.0, 2: 2.0}))
    assert len(vec[(2, 3)]) == 1
    vec[vec <= 1.0] = 1.0
    assert vec[0] == 1.0
    vec[(2, 3)] = 3.0
    assert vec[2] == vec[3] == 3.0
    del vec[vec > 1.0]
    assert len(vec) == 2
    del vec[(1, 2)]
    assert len(vec) == 1


def test_dense():
    vec = vector.fromdense(range(4))
    assert dict(vec) == {1: 1, 2: 2, 3: 3}
    assert np.array_equal(vec.todense(), np.arange(4, dtype=float))
    assert list(vec.todense(5)) == [0, 1, 2, 3, 0]


def test_math():
    vec = vector(range(3), 1.0)
    vec += 1
    assert vec.equal(vector(range(3), 2.0))
    vec -= 1
    assert vec.equal(vector(range(3), 1.0))
    vec *= 2
    assert vec.equal(vector(range(3), 2.0))
    vec **= 3
    assert vec.equal(vector(range(3), 8.0))
    vec /= 2
    assert vec.equal(vector(range(3), 4.0))

    vec += vector([3], 4.0)
    assert vec.equal(vector(range(4), 4.0))
    vec *= vector([3, 4], 2.0)
    assert vec.equal(vector([3], 8.0))
    with pytest.raises(TypeError):
        vec -= vec

    vec = vector(range(3), 1.0)
    assert (vec + 1).equal(vector(range(3), 2.0))
    assert (vec - 1).equal(vector(range(3), 0.0))
    assert (vec * 2).equal(vector(range(3), 2.0))
    assert ((vec + 1) ** 3).equal(vector(range(3), 8.0))
    assert ((vec + 1) / 2).equal(vector(range(3), 1.0))
    with pytest.raises(TypeError):
        pow(vec, 2, 2)
    assert (1 + vec).equal(vector(range(3), 2.0))
    assert (3 - vec).equal(vector(range(3), 2.0))
    assert (2 * vec).equal(vector(range(3), 2.0))
    assert (3 ** (vec + 1)).equal(vector(range(3), 9.0))
    assert (1 / (vec + 1)).equal(vector(range(3), 0.5))

    assert (vec + vector([2, 3], 2.0)).equal(vector({0: 1.0, 1: 1.0, 2: 3.0, 3: 2.0}))
    other = vector([2, 3], 2.0)
    assert (vec * other).equal(vector({2: 2.0}))
    assert (other * vec).equal(vector({2: 2.0}))
    assert vec.dot(other) == other.dot(vec) == 2.0
    try:
        eval('vec @ other') == 2.0
    except SyntaxError:
        pass


def test_ufunc():
    vec = vector({0: -1.0, 1: 0.0, 2: 1.0})
    assert (-vec).equal(vector({0: 1, 1: 0, 2: -1}))
    assert abs(vec).equal(vector({0: 1, 1: 0, 2: 1}))
    assert set(vec.map(np.minimum, 0)) == {-1.0, 0.0}
    assert set(vec.map(np.maximum, -vec)) == {0.0, 1.0}
    assert set(vec.filter(np.equal, 0.0)) == {1}
    assert set(vec.filter(np.equal, abs(vec))) == {1, 2}


def test_sets():
    vec = vector(range(3))
    other = vector({1: 0.0, 2: 2.0, 3: 1.0})
    assert (vec | other).equal(vector({0: 1.0, 1: 1.0, 2: 2.0, 3: 1.0}))
    assert (vec & other).equal(vector({1: 0.0, 2: 1.0}))
    assert not vec & vector()
    assert vec.maximum(other).equal(vector({0: 1.0, 1: 1.0, 2: 2.0}))
    assert vec.minimum(other).equal(vector({0: 0.0, 1: 0.0, 2: 1.0}))
    assert (vec ^ other).equal(vector({0: 1.0, 3: 1.0}))
    assert vec.difference(other, ()).equal(vector({0: 1.0}))

    vec |= other
    assert vec.equal(vector({0: 1.0, 1: 1.0, 2: 2.0, 3: 1.0}))
    vec &= vector({2: 1.0, 3: 2.0, 4: 1.0})
    assert vec.equal(vector({2: 1.0, 3: 1.0}))
    vec ^= other
    assert vec.equal(vector({1: 0.0}))


def test_reduce():
    vec = vector(dict(enumerate(range(5))))
    assert np.sum(vec) == 10.0
    assert np.sum(vec, dtype=int) == 10
    assert np.min(vec) == 0.0
    assert np.max(vec) == 4.0
    vec.clear()
    with pytest.raises(ValueError):
        np.min(vec)
    with pytest.raises(ValueError):
        np.max(vec)


def test_arg():
    vec = vector(dict(enumerate(range(5, 0, -1))))
    assert list(np.argsort(vec)) == [4, 3, 2, 1, 0]
    keys = np.argpartition(vec, 2)
    assert set(keys[:2]) == {3, 4} and keys[2] == 2
    assert np.argmin(vec) == 4
    assert np.argmax(vec) == 0
    with pytest.raises(ValueError):
        np.argmin(vector())
    with pytest.raises(ValueError):
        np.argmax(vector())
    arr = np.array([10, 20, 30, 20, 10])
    assert {key: list(values) for key, values in arggroupby(arr)} == {10: [0, 4], 20: [1, 3], 30: [2]}
    arr, = np.nonzero(vector({1: 0.0, 2: 1.0}))
    assert list(arr) == [2]
