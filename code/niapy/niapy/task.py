# encoding=utf8

"""The implementation of tasks."""

import logging
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from niapy.problems import Problem
from niapy.util.repair import limit
from niapy.util.factory import get_problem

logging.basicConfig()
logger = logging.getLogger("niapy.task.Task")
logger.setLevel("INFO")


class OptimizationType(Enum):
    r"""Enum representing type of optimization.

    Attributes:
        MINIMIZATION (int): Represents minimization problems and is default optimization type of all algorithms.
        MAXIMIZATION (int): Represents maximization problems.

    """

    MINIMIZATION = 1.0
    MAXIMIZATION = -1.0


class Task:
    r"""Class representing an optimization task.

    Date:
        2019

    Author:
        Klemen Berkovič and others

    Attributes:
        Name (list[str]): List of names of the problem.
        problem (Problem): Optimization problem.
        dimension (int): Dimension of the problem.
        lower (numpy.ndarray): Lower bounds of the problem.
        upper (numpy.ndarray): Upper bounds of the problem.
        range (numpy.ndarray): Search range between upper and lower limits.
        optimization_type (OptimizationType): Optimization type to use.
        iters (int): Number of algorithm iterations/generations.
        evals (int): Number of function evaluations.
        max_iters (int): Maximum number of algorithm iterations/generations.
        max_evals (int): Maximum number of function evaluations.
        cutoff_value (float): Reference function/fitness values to reach in optimization.
        x_f (float): Best found individual function/fitness value.
        x (numpy.ndarray): Best individual found.

    """
    Name = ['T', 'Task']

    def __init__(self, problem=None, optimization_type=OptimizationType.MINIMIZATION, repair_function=limit, max_evals=np.inf, max_iters=np.inf, cutoff_value=None, *args, **kwargs):
        r"""Initialize task class for optimization.

        Args:
            problem (Union[str, Problem]): Optimization problem.
            dimension (Optional[int]): Dimension of the problem. Will be ignored if problem is instance of the `Problem` class.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem. Will be ignored if problem is instance of the `Problem` class.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem. Will be ignored if problem is instance of the `Problem` class.
            optimization_type (Optional[OptimizationType]): Set the type of optimization. Default is minimization.
            repair_function (Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Dict[str, Any]], numpy.ndarray]]): Function for repairing individuals components to desired limits.
            max_evals (Optional[int]): Number of function evaluations.
            max_iters (Optional[int]): Number of maximum iterations.
            cutoff_value (Optional[float]): Reference value of function/fitness function.
            args (list): Additional args.
            kwargs (dict): Additional args.

        """
        self.problem = problem
        self.optimization_type = optimization_type
        self.dimension = self.problem.dimension
        self.lower = self.problem.lower
        self.upper = self.problem.upper
        self.range = self.upper - self.lower
        self.repair_function = repair_function

        self.evals = 0

        self.cutoff_value = -np.inf * optimization_type.value if cutoff_value is None else cutoff_value
        self.x_f = np.inf * optimization_type.value
        self.x = self.lower
        self.max_evals = max_evals
        self.max_iters = max_iters
        self.fitness_evals = []  # fitness improvements at self.n_evals evaluations

    def repair(self, x, rng=None):
        r"""Repair solution and put the solution in the random position inside of the bounds of problem.

        Args:
            x (numpy.ndarray): Solution to check and repair if needed.
            rng (Optional[numpy.random.Generator]): Random number generator.

        Returns:
            numpy.ndarray: Fixed solution.

        See Also:
            * :func:`niapy.util.repair.limit`
            * :func:`niapy.util.repair.limit_inverse`
            * :func:`niapy.util.repair.wang`
            * :func:`niapy.util.repair.rand`
            * :func:`niapy.util.repair.reflect`

        """
        return self.repair_function(x, self.lower, self.upper, rng=rng)

    def eval(self, x):
        r"""Evaluate the solution A.

        Args:
            x (numpy.ndarray): Solution to evaluate.

        Returns:
            float: Fitness/function values of solution.

        """
        if self.stopping_condition():
            return np.inf

        self.evals += 1
        x_f = self.problem.evaluate(x) * self.optimization_type.value

        if x_f < self.x_f:
            self.x_f = x_f
            self.x = np.copy(x)
            self.fitness_evals.append((self.evals, x_f))
        return x_f

    def is_feasible(self, x):
        r"""Check if the solution is feasible.

        Args:
            x (Union[numpy.ndarray, Individual]): Solution to check for feasibility.

        Returns:
            bool: `True` if solution is in feasible space else `False`.

        """
        return np.all((x >= self.lower) & (x <= self.upper))

    def stopping_condition(self):
        r"""Check if optimization task should stop.

        Returns:
            bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`.

        """
        return (self.evals >= self.max_evals) or (self.cutoff_value * self.optimization_type.value >= self.x_f * self.optimization_type.value)

    def get_best_fitness(self):
        r"""Get best individual fitness value.

        Returns:
            float: Best fitness value.

        """
        return self.x_f * self.optimization_type.value

    def get_best(self):
        r"""Get betst individual position.

        Returns:
            number.ndarray: Position of best individual.

        """
        return self.x

