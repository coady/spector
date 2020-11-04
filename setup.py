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
    version='1.1',
    description='Sparse vectors.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aric Coady',
    author_email='aric.coady@gmail.com',
    url='https://github.com/coady/spector',
    project_urls={'Documentation': 'https://coady.github.io/spector'},
    license='Apache Software License',
    packages=['spector'],
    package_data={'spector': ['py.typed']},
    zip_safe=False,
    ext_modules=cythonize([ext_module]) if cythonize else [ext_module],
    install_requires=['numpy'],
    python_requires='>=3.6',
    tests_require=['pytest-cov'],
    keywords='sparse array vector matrix numpy scipy',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Typing :: Typed',
    ],
)
