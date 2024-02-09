from setuptools import Extension
from distutils.core import setup
from setuptools.command.test import test as TestCommand
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import sys

# cython = Extension('ea.cbenchmarks',
#    sources = ['ea/cbenchmarks.pyx'],
#    include_dirs = ['include/']
# )
sourcefiles = ['cec2013lsgo/cec2013.pyx']

sourcefiles += ['cec2013lsgo/eval_func.cpp', 'cec2013lsgo/Benchmarks.cpp']

for i in range(1, 16):
    sourcefiles += ['cec2013lsgo/F%d.cpp' % i]

cec2013lsgo = Extension("cec2013lsgo.cec2013",
                        sourcefiles,
                        language="c++",
                        extra_compile_args=["-std=c++11", "-O3"],
                        libraries=["m"])  # Unix-like specific


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self): 
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    cmdclass={'build_ext': build_ext, 'test': PyTest},
    ext_modules=cythonize(cec2013lsgo),
    packages=['cec2013lsgo'],
    package_data={'cec2013lsgo': ['cdatafiles/*.txt']},
    tests_require=['pytest'],
)
