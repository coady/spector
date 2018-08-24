all:
	cythonize spector/*.pyx
	python3 setup.py build_ext -i

check: all
	python3 setup.py $@ -ms
	flake8
	pytest --cov

html:
	cythonize -aX linetrace=True spector/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE
	pytest --cov --cov-report $@

dist: all
	python3 setup.py sdist bdist_wheel
	docker run --rm -v $(PWD):/usr/src -w /usr/src python python setup.py bdist_wheel

clean:
	hg st -in | xargs rm
	rm -rf build dist spector.egg-info htmlcov
	rm -f spector/*.html
