from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

ext_module = Extension(
    'spector.vector',
    sources=['spector/vector' + ('.pyx' if cythonize else '.cpp')],
    extra_compile_args=['-std=c++11'],
    extra_link_args=['-std=c++11'],
)

setup(ext_modules=cythonize([ext_module]) if cythonize else [ext_module])
