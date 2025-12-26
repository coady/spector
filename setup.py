from setuptools import Extension, setup

ext_module = Extension("spector.vector", sources=["spector/vector.cpp"])
setup(ext_modules=[ext_module])
