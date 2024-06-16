all: spector/*.pyx
	cythonize -aX linetrace=True $?
	python setup.py build_ext -i --define CYTHON_TRACE_NOGIL

check: all
	python -m pytest -s --cov

bench: spector/*.pyx
	cythonize $?
	python setup.py build_ext -i
	python -m pytest --codspeed

lint:
	ruff check .
	ruff format --check .
	cython-lint spector/*.pyx --ignore E501
	mypy -p spector

html: all
	PYTHONPATH=$(PWD) python -m mkdocs build
