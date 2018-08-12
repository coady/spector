from setuptools import setup, Extension
from Cython.Build import cythonize

ext_module = Extension('spector.vector',
                       sources=['spector/vector.pyx'],
                       extra_compile_args=['-std=c++11'],
                       extra_link_args=['-std=c++11'])

setup(
    name='spector',
    version='0.0',
    description='Sparse vectors.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aric Coady',
    author_email='aric.coady@gmail.com',
    url='https://bitbucket.org/coady/spector',
    license='Apache Software License',
    packages=['spector'],
    ext_modules=cythonize([ext_module], language='c++'),
    keywords='sparse array vector matrix numpy scipy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
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
