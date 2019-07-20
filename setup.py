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

setup(
    name='spector',
    version='0.2',
    description='Sparse vectors.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aric Coady',
    author_email='aric.coady@gmail.com',
    url='https://github.com/coady/spector',
    project_urls={'Documentation': 'https://spector.readthedocs.io'},
    license='Apache Software License',
    packages=['spector'],
    ext_modules=cythonize([ext_module]) if cythonize else [ext_module],
    install_requires=['numpy'],
    extras_require={'docs': ['m2r', 'nbsphinx', 'jupyter', 'pandas']},
    python_requires='>=2.7',
    tests_require=['pytest-cov'],
    keywords='sparse array vector matrix numpy scipy',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
