all:
	python -m cython -a -X linetrace=True --cplus spector/*.pyx
	python setup.py build_ext -i --define CYTHON_TRACE_NOGIL

check: all
	pytest -s --cov

lint:
	black --check .
	ruff .
	flake8 spector/*.pyx --ignore E999
	mypy -p spector

html: all
	PYTHONPATH=$(PWD) python -m mkdocs build
