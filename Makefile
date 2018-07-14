all:
	python3 setup.py build_ext -i

check: all
	python3 setup.py $@ -mrs
	flake8
	pytest --cov

clean:
	hg st -in | xargs rm
	rm -rf build dist spector.egg-info
