# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import Task
from niapy.problems import Griewank

from niapy.algorithms.analysis import (
    RecursiveDifferentialGroupingV3,
    RecursiveDifferentialGroupingV2,
    ExtendedDifferentialGrouping
)

algo = RecursiveDifferentialGroupingV2(seed=1)
algo.set_parameters(n=50)
for i in range(5):
    task = Task(problem=Griewank(dimension=10, lower=-600, upper=600), max_evals=10000, enable_logging=True)
    best = algo.run(task)
    print (best)
print(algo.get_parameters())

algo = RecursiveDifferentialGroupingV3(seed=1)
algo.set_parameters(n=50)
for i in range(5):
    task = Task(problem=Griewank(dimension=10, lower=-600, upper=600), max_evals=10000, enable_logging=True)
    best = algo.run(task)
    print (best)
print(algo.get_parameters())

algo = ExtendedDifferentialGrouping(seed=1)
algo.set_parameters(n=50)
for i in range(5):
    task = Task(problem=Griewank(dimension=10, lower=-600, upper=600), max_evals=10000, enable_logging=True)
    best = algo.run(task)
    print (best)
print(algo.get_parameters())
