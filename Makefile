all:
	cythonize -aX linetrace=True spector/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE_NOGIL

check: all
	pytest -s --cov

lint:
	black --check .
	flake8
	flake8 spector/*.pyx --ignore E999
	mypy -p spector

html: all
	PYTHONPATH=$(PWD) python3 -m mkdocs build
