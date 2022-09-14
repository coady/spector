[![image](https://img.shields.io/pypi/v/spector.svg)](https://pypi.org/project/spector/)
[![image](https://img.shields.io/pypi/pyversions/spector.svg)](https://python3statement.org)
[![image](https://pepy.tech/badge/spector)](https://pepy.tech/project/spector)
![image](https://img.shields.io/pypi/status/spector.svg)
[![image](https://github.com/coady/spector/workflows/build/badge.svg)](https://github.com/coady/spector/actions)
[![image](https://codecov.io/gh/coady/spector/branch/main/graph/badge.svg)](https://codecov.io/gh/coady/spector/)
 [![image](https://github.com/coady/spector/workflows/codeql/badge.svg)](https://github.com/coady/spector/security/code-scanning)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
[![image](http://mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

Sparse vectors optimized for memory and [NumPy](http://www.numpy.org) integrations.

`numpy` handles densely populated n-dimemsional arrays. `scipy.sparse` handles sparsely populated 2-dimensional arrays, i.e., matrices. What's missing from the ecosystem is sparsely populated 1-dimensional arrays, i.e., vectors.

NumPy | Python | Spector
----- | ------ | -------
1-dim bool `numpy.array` | `set[int]` | `spector.indices`
1-dim float `numpy.array` | `dict[int, float]` | `spector.vector`
`scipy.sparse.dok_matrix` | `dict[int, dict[int, float]]` | `spector.matrix`

Indices and vectors are implemented in [Cython](https://cython.org) as hash sets and maps. All native operations are optimized and release the [GIL](https://docs.python.org/3/glossary.html#term-global-interpreter-lock).

* conversion between sparse `numpy` arrays
* conversion between dense `numpy` arrays
* binary set operations
* binary math operations
* `map`, `filter`, and `reduce` operations with `numpy` universal functions

## Usage
### indices
A sparse boolean array with a set interface.

```python
>>> from spector import indices
>>> ind = indices([0, 2])
>>> ind
indices([2 0])
>>> 1 in ind
False
>>> ind.add(1)
True
>>> ind.todense()
array([ True,  True,  True])
>>> ind.fromdense(_)
indices([2 1 0])
```

### vector
A sparse float array with a mapping interface.

```python
>>> from spector import vector
>>> vec = vector({0: 1.0, 2: 2.0, 4: 1.0})
>>> vec
vector([4 2 0], [1. 2. 1.])
>>> vec[2] += 1.0
>>> vec[2]
3.0
>>> vec.todense()
array([1., 0., 3., 0., 1.])
>>> vector.fromdense(_)
vector([4 2 0], [1. 3. 1.])
>>> vec.sum()
5.0
>>> vec + vec
vector([0 2 4], [2. 6. 2.])
```

Vectors support math operations with scalars, and with vectors if the set method is unambiguous.

vector operation | set method | ufunc
---------------- | ---------- | -----
`+` | union | add
`*` | intersection | multiply
`-` | | subtract
`/` | | true_divide
`**` | | power
`\|` | union | max
`&` | intersection | min
`^` | symmetric_difference |
`difference` | difference |

### matrix
A mapping of keys to vectors.

```python
>>> from spector import matrix
>>> mat = matrix({0: {1: 2.0}})
>>> mat
matrix(<class 'spector.vector.vector'>, {0: vector([1], [2.])})
>>> mat.row, mat.col, mat.data
(array([0]), array([1]), array([2.]))
```

## Installation
```console
% pip install spector
```

## Tests
100% branch coverage.

```console
% pytest [--cov]
```

## Changes
1.3

* Python 3.11 wheels
* Cython updates

1.2

* Python >=3.7 required

1.1

* Python >=3.6 required
* Read-only views supported
* Optimized intersection count

1.0

* Optimized set intersection

0.2

* Numerous optimizations
* Intersection count
* Increased parallelism
