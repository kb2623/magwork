#!python
#cython: language_level=2, boundscheck=False
import os
import sys
import pkgutil

from collections import namedtuple

import cython
from libc.stdlib cimport malloc, free


if sys.version < '3':
    def b(x):
        return x
else:
    import codecs
    def b(x):
        return codecs.latin_1_encode(x)[0]


cdef extern from "eval_func.h":
    void set_func(int funid)
    double eval_sol(double*)
    void set_data_dir(char * new_data_dir)
    void free_func()
    void next_run()


def _cec2013_test_func(double[::1] x):
    cdef int dim
    cdef double fitness
    cdef double * sol

    dim = x.shape[0]

    sol = <double *> malloc(dim * cython.sizeof(double))

    if sol is NULL:
        raise MemoryError()

    for i in xrange(dim):
        sol[i] = x[i]

    fitness = eval_sol(sol)
    free(sol)
    return fitness


def file_load(data_dir: str, file_name: str):
    """Load cdata file into temporary directory.

    Args:
        data_dir (str): Path to storage of data files
        file_name (str): File name to load.
    """
    if os.path.exists('%s/%s' % (data_dir, file_name)): return
    data = pkgutil.get_data('cec2013lsgo', 'cdatafiles/%s' % file_name)
    with open('%s/%s' % (data_dir, file_name), 'wb') as f: f.write(data)


cdef class Benchmark:
    cdef public str input_data_dir

    def __init__(self, input_data_dir: str = 'inputdata'):
        # Set input data dir
        self.input_data_dir = input_data_dir
        # Create input data dir
        os.makedirs(self.input_data_dir, exist_ok=True)
        # Load xopt
        for i in range(1, 16): file_load(self.input_data_dir, 'F%d-xopt.txt' % i)
        # Load p
        for i in range(4, 12): file_load(self.input_data_dir, 'F%d-p.txt' % i)
        for i in range(13, 15): file_load(self.input_data_dir, 'F%d-p.txt' % i)
        # Load w
        for i in range(4, 12): file_load(self.input_data_dir, 'F%d-w.txt' % i)
        for i in range(13, 15): file_load(self.input_data_dir, 'F%d-w.txt' % i)
        # Load s
        for i in range(4, 12): file_load(self.input_data_dir, 'F%d-s.txt' % i)
        for i in range(13, 15): file_load(self.input_data_dir, 'F%d-s.txt' % i)
        # Load R25
        for i in range(4, 12): file_load(self.input_data_dir, 'F%d-R25.txt' % i)
        for i in range(13, 15): file_load(self.input_data_dir, 'F%d-R25.txt' % i)
        # Load R50
        for i in range(4, 12): file_load(self.input_data_dir, 'F%d-R50.txt' % i)
        for i in range(13, 15): file_load(self.input_data_dir, 'F%d-R50.txt' % i)
        # Load R100
        for i in range(4, 12): file_load(self.input_data_dir, 'F%d-R100.txt' % i)
        for i in range(13, 15): file_load(self.input_data_dir, 'F%d-R100.txt' % i)
        
    cpdef get_info(self, int fun):
        r"""Return the lower bound of the function.

        Args:
            fun (int): Optimization function number.
        """
        cdef double optimum
        cdef double range_fun

        optimum = 0

        if (fun in [2, 5, 9]):
            range_fun = 5
        elif (fun in [3, 6, 10]):
            range_fun = 32
        else:
            range_fun = 100

        return {'lower': -range_fun, 'upper': range_fun, 'threshold': 0,
                'best': optimum, 'dimension': 1000}

    def get_num_functions(self):
        return 15

    def __dealloc(self):
        free_func()

    cpdef next_run(self):
        next_run()

    cpdef get_function(self, int fun):
        r"""Get optimization function for evaluate the solution.

        Args:
            fun (int): Optimization fucntion number.
        """
        set_func(fun)
        cdef bytes dir_name = ('%s/%s' % (os.getcwd(), self.input_data_dir)).encode()
        set_data_dir(dir_name)
        return _cec2013_test_func
    
