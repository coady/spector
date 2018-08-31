all:
	cythonize spector/*.pyx
	python2 setup.py build_ext -i
	python3 setup.py build_ext -i

check: all
	python3 setup.py $@ -ms
	flake8
	pytest-2.7
	pytest --cov

html:
	cythonize -aX linetrace=True spector/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE --define CYTHON_TRACE_NOGIL
	pytest --cov --cov-report $@

dist: all
	python3 setup.py sdist bdist_wheel
	docker run --rm -v $(PWD):/usr/src -w /usr/src python python setup.py bdist_wheel

clean:
	hg st -in | xargs rm
	rm -rf build dist spector.egg-info htmlcov
	rm -f spector/*.html
