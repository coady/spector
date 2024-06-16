import random
import numpy as np
from spector import indices, vector

size = 10**5


def keys():
    return np.array([random.randint(0, size * 2) for _ in range(size)])


def test_indices(benchmark):
    inds = benchmark(indices, keys()), indices(keys())
    benchmark(indices.__matmul__, *inds)
    benchmark(indices.__and__, *inds)
    benchmark(indices.__or__, *inds)


def test_vector(benchmark):
    vecs = benchmark(vector, keys()), vector(keys())
    benchmark(vector.__matmul__, *vecs)
    benchmark(vector.__mul__, *vecs)
    benchmark(vector.__add__, *vecs)
