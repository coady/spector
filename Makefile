all: spector/*.cpp
	python setup.py build_ext -i --define CYTHON_TRACE_NOGIL

spector/*.cpp: spector/*.pyx
	python -m cython -aX linetrace=True --cplus $?

check: all
	python -m pytest -s --cov

lint:
	ruff .
	ruff format --check .
	cython-lint spector/*.pyx --ignore E501
	mypy -p spector

html: all
	PYTHONPATH=$(PWD) python -m mkdocs build
