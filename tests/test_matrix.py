import numpy as np
from spector import matrix


def test_basic():
    mat = matrix()
    mat[0][1] = 2.0
    assert np.array_equal(mat.row, np.array([0]))
    assert np.array_equal(mat.col, np.array([1]))
    assert np.array_equal(mat.data, np.array([2.0]))
