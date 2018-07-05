import numpy as np
import pytest
from spector import vector


def test_defaults():
    vec = vector(range(3))
    assert vec.dtype.name == 'int64'
    assert str(vec) == 'vector([2 1 0], [1 1 1])'
    assert len(vec) == 3
    assert np.array_equal(vec.keys(), np.array([2, 1, 0]))
    assert np.array_equal(vec.values(), np.array([1, 1, 1]))
    assert vec[0] == 1
    vec[0] += 1.5
    assert vec[0] == 2
    assert 0 in vec
    del vec[0]
    assert 0 not in vec
    assert vec[0] == 0
    assert 0 in vec
    with pytest.raises(TypeError):
        vec[None]
    assert set(vec) == {0, 1, 2}
    assert dict(vec.items()) == {0: 0, 1: 1, 2: 1}
    vec.clear()
    assert not vec


def test_types():
    vec = vector(range(3), 1.0)
    assert np.array_equal(vec.values(), np.array([1.0, 1.0, 1.0]))
    vec = vector(range(3), [0, 5], dtype=int)
    assert np.array_equal(vec.values(), np.array([5, 0]))
    vec = vector(range(3), dtype=float)
    assert np.array_equal(vec.values(), np.array([1.0, 1.0, 1.0]))
    vec = vector(range(3), np.array([0, 5, 10]))
    assert np.array_equal(vec.values(), np.array([10, 5, 0]))
    vec.update(vector(range(3, 6)))
    assert sorted(vec.keys()) == list(range(6))
