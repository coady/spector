all:
	cythonize -aX linetrace=True spector/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE_NOGIL
	python2 setup.py build_ext -i

check: all
	python3 setup.py $@ -ms
	black -q --check .
	flake8
	flake8 spector/*.pyx --ignore E999,E211,E225
	pytest-2.7
	pytest --cov --cov-fail-under=100

html: all
	pytest --cov --cov-report $@
	make -C docs $@

dist:
	python3 setup.py sdist bdist_wheel
	docker run --rm -v $(PWD):/usr/src -w /usr/src quay.io/pypa/manylinux1_x86_64 make cp35 cp36 cp37

cp35 cp36 cp37:
	/opt/python/$@-$@m/bin/pip wheel . -w dist
	auditwheel repair dist/*$@m-linux_x86_64.whl

clean:
	hg st -in | xargs rm
	rm -rf build dist spector.egg-info htmlcov wheelhouse
	rm -f spector/*.html
