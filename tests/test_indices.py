import numpy as np
import pytest
from spector import indices


def test_basic():
    ind = indices()
    assert str(ind) == 'indices([])'
    assert len(ind) == 0
    assert 0 not in ind
    assert list(ind) == []
    assert indices(np.array([False]))
    with pytest.raises(TypeError):
        indices(np.array([0.0]))

    assert ind.add(0)
    assert not ind.add(0)
    assert str(ind) == 'indices([0])'
    assert len(ind) == 1
    assert 0 in ind
    assert list(ind) == [0]

    assert ind.discard(0)
    assert not ind.discard(0)
    ind.update(range(3))
    assert set(ind) == set(indices(np.array(ind))) == {0, 1, 2}
    np.array(ind).sum() == 3
    ind.update(indices([3]))
    assert 3 in ind
    ind.clear()
    assert not ind


def test_cmp():
    ind = indices([0, 1])
    assert ind == ind == indices([0, 1]) == indices([1, 0])
    assert ind != {} and not (ind == {})
    assert not (ind != ind)
    assert ind != indices([0, 2])
    assert not (ind <= indices([0]))
    assert ind <= ind
    assert not ind < ind
    assert ind < indices([0, 1, 2])
    with pytest.raises(TypeError):
        ind < None
    with pytest.raises(TypeError):
        ind <= None
    assert not ind.isdisjoint(indices([0])) and ind.isdisjoint(indices([2]))


def test_sets():
    x, y = indices([0, 1]), indices([1, 2])
    assert x | y == indices([0, 1, 2])
    assert x & y == indices([1])
    assert x - y == indices([0])
    assert x ^ y == indices([0, 2])

    z = x
    z ^= y
    assert z is x and z == indices([0, 2])
    z |= y
    assert z is x and z == indices([0, 1, 2])
    z -= y
    z -= y  # smaller vector is iterated
    assert z is x and z == indices([0])
    z &= y
    assert z is x and z == indices()
    assert not y & z


def test_dense():
    arr = np.zeros(4, bool)
    arr[0] = arr[2] = True
    ind = indices.fromdense(arr)
    assert set(ind) == {0, 2}
    assert np.array_equal(ind.todense(), arr[:3])
    assert np.array_equal(ind.todense(4), arr)
    with pytest.raises(IndexError):
        ind.todense(2)
