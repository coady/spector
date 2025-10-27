import collections
import random

import numpy as np
import pytest

from spector import groupby, indices, vector

size = 10**5


def keys():
    return np.array([random.randint(0, size * 2) for _ in range(size)])


keys1 = pytest.fixture(keys)
keys2 = pytest.fixture(keys)


@pytest.mark.benchmark
def test_indices(keys1, keys2):
    i, j = indices(keys1), indices(keys2)
    i @ j
    i & j
    i | j


@pytest.mark.benchmark
def test_vector(keys1, keys2):
    v, w = vector(keys1), vector(keys2)
    v @ w
    v * w
    v + w


@pytest.mark.benchmark
def test_group_hashed(keys1):
    collections.deque(groupby(keys1), maxlen=0)


@pytest.mark.benchmark
def test_group_sorted(keys1):
    collections.deque(groupby(keys1.astype('u8')), maxlen=0)
