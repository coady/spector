[![image](https://img.shields.io/pypi/v/spector.svg)](https://pypi.org/project/spector/)
![image](https://img.shields.io/pypi/pyversions/spector.svg)
![image](https://img.shields.io/pypi/status/spector.svg)
[![image](https://img.shields.io/travis/coady/spector.svg)](https://travis-ci.org/coady/spector)
[![image](https://img.shields.io/codecov/c/github/coady/spector.svg)](https://codecov.io/github/coady/spector)
[![image](https://requires.io/github/coady/spector/requirements.svg)](https://requires.io/github/coady/spector/requirements/)
[![image](https://api.codeclimate.com/v1/badges/6ffbd68facb9ef4acfef/maintainability)](https://codeclimate.com/github/coady/spector/maintainability)

Sparse vectors optimized for memory and [NumPy](http://www.numpy.org) integrations.

NumPy | Python | Spector
----- | ------ | -------
1-dim bool `numpy.array` | `set` | `spector.indices`
1-dim float `numpy.array` | `dict` | `spector.vector`
`scipy.sparse.dok_matrix` | `dict` | `spector.matrix`

# Usage
```python
from spector import indices, vector, matrix

>>> ind = indices(range(3))
>>> ind.add(3)
True
>>> ind.discard(0)
1
>>> np.array(ind)
array([3, 2, 1])
>>> ind.todense(5)
array([False,  True,  True,  True, False])

>>> vec = vector(range(3))
>>> del vec[0]
>>> vec[2] += 1.0
>>> vec[3] += 1.0
>>> vec.keys()
array([3, 2, 1])
>>> vec.values()
array([1., 2., 1.])
>>> np.array(vec)
array([1., 2., 1.])
>>> vec.todense(5)
array([0., 1., 2., 1., 0.])
```

# Installation
    $ pip install spector

# Tests
100% branch coverage.

    $ pytest [--cov]
