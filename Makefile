all:
	cythonize -aX linetrace=True spector/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE_NOGIL

check: all
	pytest --cov

lint:
	python3 setup.py check -ms
	black --check .
	flake8
	flake8 spector/*.pyx --ignore E999,E211,E225
	mypy -p spector

html: all
	PYTHONPATH=$(PWD) mkdocs build

dist:
	python3 setup.py sdist bdist_wheel
	docker run --rm -v $(PWD):/usr/src -w /usr/src quay.io/pypa/manylinux_2_24_x86_64 make cp36 cp37 cp38 cp39

cp36 cp37:
	/opt/python/$@-$@m/bin/pip install cython
	/opt/python/$@-$@m/bin/python setup.py build
	/opt/python/$@-$@m/bin/pip wheel . -w dist
	auditwheel repair dist/*$@m-linux_x86_64.whl

cp38 cp39:
	/opt/python/$@-$@/bin/pip install cython
	/opt/python/$@-$@/bin/python setup.py build
	/opt/python/$@-$@/bin/pip wheel . -w dist
	auditwheel repair dist/*$@-linux_x86_64.whl
