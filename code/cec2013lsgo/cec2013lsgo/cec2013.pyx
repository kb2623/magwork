#!python
#cython: language_level=2, boundscheck=False
import pkgutil
import tempfile
from os import path
from collections import namedtuple
from libc.stdlib cimport malloc, free
import cython

cdef extern from "eval_func.h":
    void set_func(int funid)
    double eval_sol(double*)
    void set_data_dir(char * new_data_dir)
    void free_func()
    void next_run()


import sys
if sys.version < '3':
    def b(x):
        return x
else:
    import codecs
    def b(x):
        return codecs.latin_1_encode(x)[0]


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


cdef class Benchmark:
    def __init__(self):
        """Init Benchmark and create files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        cdef bytes dir_name = self.temp_dir.name.encode()
        set_data_dir(dir_name)
        
    cpdef get_info(self, int fun):
        """Return the lower bound of the function.

        Args:
            fun (int): Optimization funciton number.

        Returns:
            dict[str, int]: Information about the optmization funciton.
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
        # Delete temprary directory with cdatafiles
        self.temp_dir.cleanup()
        # Clean memory of the functions
        free_func()

    cpdef next_run(self):
        next_run()

    def __file_load(self, file_name):
        """Load cdata file into temporary directory.

        Args:
            file_name (str): File name to load.
        """
        if path.exists('%s/%s' % (self.temp_dir.name, file_name)): return
        data = pkgutil.get_data('cec2013lsgo', 'cdatafiles/%s' % file_name)
        with open('%s/%s' % (self.temp_dir.name, file_name), 'wb') as f: f.write(data)

    def __load_f1(self):
        self.__file_load('F1-xopt.txt')

    def __load_f2(self):
        self.__file_load('F2-xopt.txt')

    def __load_f3(self):
        self.__file_load('F3-xopt.txt')

    def __load_f4(self):
        self.__file_load('F4-xopt.txt')
        self.__file_load('F4-p.txt')
        self.__file_load('F4-w.txt')
        self.__file_load('F4-s.txt')
        self.__file_load('F4-R25.txt')
        self.__file_load('F4-R50.txt')
        self.__file_load('F4-R100.txt')

    def __load_f5(self):
        self.__file_load('F5-xopt.txt')
        self.__file_load('F5-p.txt')
        self.__file_load('F5-w.txt')
        self.__file_load('F5-s.txt')
        self.__file_load('F5-R25.txt')
        self.__file_load('F5-R50.txt')
        self.__file_load('F5-R100.txt')

    def __load_f6(self):
        self.__file_load('F6-xopt.txt')
        self.__file_load('F6-p.txt')
        self.__file_load('F6-w.txt')
        self.__file_load('F6-s.txt')
        self.__file_load('F6-R25.txt')
        self.__file_load('F6-R50.txt')
        self.__file_load('F6-R100.txt')

    def __load_f7(self):
        self.__file_load('F7-xopt.txt')
        self.__file_load('F7-p.txt')
        self.__file_load('F7-w.txt')
        self.__file_load('F7-s.txt')
        self.__file_load('F7-R25.txt')
        self.__file_load('F7-R50.txt')
        self.__file_load('F7-R100.txt')

    def __load_f8(self):
        self.__file_load('F8-xopt.txt')
        self.__file_load('F8-p.txt')
        self.__file_load('F8-w.txt')
        self.__file_load('F8-s.txt')
        self.__file_load('F8-R25.txt')
        self.__file_load('F8-R50.txt')
        self.__file_load('F8-R100.txt')

    def __load_f9(self):
        self.__file_load('F9-xopt.txt')
        self.__file_load('F9-p.txt')
        self.__file_load('F9-w.txt')
        self.__file_load('F9-s.txt')
        self.__file_load('F9-R25.txt')
        self.__file_load('F9-R50.txt')
        self.__file_load('F9-R100.txt')

    def __load_f10(self):
        self.__file_load('F10-xopt.txt')
        self.__file_load('F10-p.txt')
        self.__file_load('F10-w.txt')
        self.__file_load('F10-s.txt')
        self.__file_load('F10-R25.txt')
        self.__file_load('F10-R50.txt')
        self.__file_load('F10-R100.txt')

    def __load_f11(self):
        self.__file_load('F11-xopt.txt')
        self.__file_load('F11-p.txt')
        self.__file_load('F11-w.txt')
        self.__file_load('F11-s.txt')
        self.__file_load('F11-R25.txt')
        self.__file_load('F11-R50.txt')
        self.__file_load('F11-R100.txt')

    def __load_f12(self):
        self.__file_load('F12-xopt.txt')

    def __load_f13(self):
        self.__file_load('F13-xopt.txt')
        self.__file_load('F13-p.txt')
        self.__file_load('F13-w.txt')
        self.__file_load('F13-s.txt')
        self.__file_load('F13-R25.txt')
        self.__file_load('F13-R50.txt')
        self.__file_load('F13-R100.txt')

    def __load_f14(self):
        self.__file_load('F14-xopt.txt')
        self.__file_load('F14-p.txt')
        self.__file_load('F14-w.txt')
        self.__file_load('F14-s.txt')
        self.__file_load('F14-R25.txt')
        self.__file_load('F14-R50.txt')
        self.__file_load('F14-R100.txt')

    def __load_f15(self):
        self.__file_load('F15-xopt.txt')

    cpdef get_function(self, int fun):
        """Get the optimization funciton and load other needed files.

        Args:
            fun (int): Optimization function number.

        Returns:
            Callable[[list[float]], float]: Optimization function.
        """
        set_func(fun)
        if fun == 1: self.__load_f1()
        elif fun == 2: self.__load_f2()
        elif fun == 3: self.__load_f3()
        elif fun == 4: self.__load_f4()
        elif fun == 5: self.__load_f5()
        elif fun == 6: self.__load_f6()
        elif fun == 7: self.__load_f7()
        elif fun == 8: self.__load_f8()
        elif fun == 9: self.__load_f9()
        elif fun == 10: self.__load_f10()
        elif fun == 11: self.__load_f11()
        elif fun == 12: self.__load_f12()
        elif fun == 13: self.__load_f13()
        elif fun == 14: self.__load_f14()
        elif fun == 15: self.__load_f15()
        return _cec2013_test_func

