all:
	cythonize -aX linetrace=True spector/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE_NOGIL

check: all
	python3 setup.py $@ -ms
	black -q --check .
	flake8
	flake8 spector/*.pyx --ignore E999,E211,E225
	mypy -p spector
	pytest --cov --cov-fail-under=100

html: all
	make -C docs $@ SPHINXOPTS=-W

dist:
	python3 setup.py sdist bdist_wheel
	docker run --rm -v $(PWD):/usr/src -w /usr/src quay.io/pypa/manylinux2014_x86_64 make cp35 cp36 cp37 cp38

cp35 cp36 cp37:
	/opt/python/$@-$@m/bin/pip wheel . -w dist
	auditwheel repair dist/*$@m-linux_x86_64.whl

cp38:
	/opt/python/$@-$@/bin/pip wheel . -w dist
	auditwheel repair dist/*$@-linux_x86_64.whl
