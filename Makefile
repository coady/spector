all:
	python3 setup.py build_ext -i

check: all
	python3 setup.py $@ -ms
	flake8
	pytest --cov

html: spector/vector.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE
	pytest --cov --cov-report $@
	cython -a $?

clean:
	hg st -in | xargs rm
	rm -rf build dist spector.egg-info htmlcov
	rm -f spector/*.c spector/*.html
