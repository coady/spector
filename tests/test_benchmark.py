import resource
import time
from concurrent import futures
from functools import partial
import numpy as np
from spector import indices, vector


def memory(unit=1e6):
    """Return memory usage in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / unit


def diff(metric, func, *args):
    """Return metric difference before and after function call."""
    start = metric()
    _ = func(*args)  # noqa
    return metric() - start


def sized(func, *args):
    """Measure memory in a separate process."""
    return executor.submit(diff, memory, func, *args).result()


timed = partial(diff, time.time)
executor = futures.ProcessPoolExecutor()
keys = np.array(range(2 ** 19))
values = np.ones(len(keys))


def test_set():
    print('set')
    print('memory', sized(indices, keys) / sized(set, keys))
    print('new', timed(indices, keys) / timed(set, keys))
    c, py = indices(keys), set(keys)
    print('array', timed(np.array, c) / timed(np.fromiter, py, keys.dtype, len(py)))


def test_vector():
    print('vector')
    print('memory', sized(vector, keys, values) / sized(set, zip(keys, values)))
    print('new', timed(vector, keys, values) / timed(set, zip(keys, values)))
    c, py = vector(keys, values), dict(zip(keys, values))
    print('keys', timed(c.keys) / timed(np.fromiter, py.keys(), keys.dtype, len(py)))
    print('values', timed(c.values) / timed(np.fromiter, py.values(), values.dtype, len(py)))
