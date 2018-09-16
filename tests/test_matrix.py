import numpy as np
import pytest
from spector import matrix, vector


def test_basic():
    mat = matrix({'a': [0], 'b': [1]})
    assert np.array_equal(np.sort(mat.row), np.array(['a', 'b']))
    assert np.array_equal(np.sort(mat.col), np.array([0, 1]))
    assert np.array_equal(mat.data, np.array([1.0, 1.0]))
    mat.update([('b', [1]), ('c', [2])])
    assert dict(zip(zip(mat.row, mat.col), mat.data)) == {('a', 0): 1.0, ('b', 1): 2.0, ('c', 2): 1.0}


def test_math():
    mat = matrix({'a': [0], 'b': [1]})
    assert mat.sum() == 2.0
    mat += matrix({'b': [1], 'c': [2]})
    assert mat.sum(1) == {'a': 1.0, 'b': 2.0, 'c': 1.0}
    assert (mat + 1).sum(-1) == {'a': 2.0, 'b': 3.0, 'c': 2.0}
    vec = vector({0: 1.0, 1: 2.0, 2: 1.0})
    assert mat.sum(0).equal(vec)
    assert mat.sum(-2).equal(vec)
    with pytest.raises(ValueError):
        mat.sum(axis=2)

    mat = matrix({'a': [0], 'b': [1]})
    other = matrix({'b': {1: 2.0}, 'c': [2]})
    (key, vec), = (mat * other).items()
    assert key == 'b' and vec.equal(vector({1: 2.0}))
    mat *= other
    assert list(mat) == ['b'] and vec.equal(*mat.values())
    (key, vec), = (mat * 2).items()
    assert vec.values() == np.array([4.0])
    mat *= 2
    assert vec.equal(*mat.values())


def test_funcs():
    mat = matrix({'a': [0]})
    assert mat.map(np.sum, dtype=int) == {'a': 1}
    assert mat.filter(len) == mat
    assert not mat.filter(vector.dot, vector())
