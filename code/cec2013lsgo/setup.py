import sys

from setuptools import (
    find_packages,
    Extension,
    setup
)

from Cython.Distutils import build_ext
from Cython.Build import cythonize


sourcefiles = [
    'cec2013lsgo/cec2013.pyx', 
    'cec2013lsgo/eval_func.cpp',
    'cec2013lsgo/Benchmarks.cpp',
]


for i in range(1, 16): 
    sourcefiles.append('cec2013lsgo/F%d.cpp' % i)

cec2013lsgo = Extension(
    "cec2013lsgo.cec2013",
    sources=sourcefiles,
    include_dirs=['cec2013lsgo/'],
    language="c++",
    extra_compile_args=["-std=c++11", "-O3"],
    libraries=["m"]
)

setup(
    ext_modules=[cec2013lsgo],
    packages=find_packages(),
    package_data={'cec2013lsgo': ['cdatafiles/*.txt', '*.h']},
)
