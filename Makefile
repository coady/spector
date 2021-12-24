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

dist:
	python3 -m build -n
	docker run --rm -v $(PWD):/usr/src -w /usr/src quay.io/pypa/manylinux_2_24_x86_64 make cp37 cp38 cp39 cp310

cp37:
	/opt/python/$@-$@m/bin/pip install cython
	/opt/python/$@-$@m/bin/python -m build -nw
	auditwheel repair dist/*$@m-linux_x86_64.whl

cp38 cp39 cp310:
	/opt/python/$@-$@/bin/pip install cython
	/opt/python/$@-$@/bin/python -m build -nw
	auditwheel repair dist/*$@-linux_x86_64.whl
