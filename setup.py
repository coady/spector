from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='spector',
    version='0.0',
    description='Sparse vectors.',
    author='Aric Coady',
    author_email='aric.coady@gmail.com',
    url='https://bitbucket.org/coady/spector',
    license='Apache Software License',
    packages=['spector'],
    ext_modules=cythonize('spector/vector.pyx', language='c++'),
    keywords='sparse array vector matrix numpy scipy',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
