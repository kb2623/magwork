# encoding=utf8

"""Module with implementations of basic and hybrid algorithms."""

from niapy.algorithms import basic
from niapy.algorithms import ccbasic
from niapy.algorithms import modified
from niapy.algorithms import other
from niapy.algorithms import analysis
from niapy.algorithms.algorithm import AnalysisAlgorithm, OptimizationAlgorithm, CoEvolutionOptimizationAlgorithm, Individual, default_numpy_init, default_individual_init

__all__ = [
    'basic',
    'ccbasic',
    'modified',
    'other',
    'AnalysisAlgorithm',
    'OptimizationAlgorithm',
    'CoEvolutionOptimizationAlgorithm',
    'default_numpy_init',
    'default_individual_init',
    'Individual',
]
