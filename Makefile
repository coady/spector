all:
	python3 -m cython -a -X linetrace=True --cplus spector/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE_NOGIL

check: all
	pytest -s --cov

lint:
	black --check .
	flake8 --ignore E501 spector tests
	flake8 spector/*.pyx --ignore E999
	mypy -p spector

html: all
	PYTHONPATH=$(PWD) python3 -m mkdocs build
