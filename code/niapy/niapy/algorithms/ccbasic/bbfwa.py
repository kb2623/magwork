# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import CoEvolutionOptimizationAlgorithm
from niapy.util.distances import euclidean
import niapy.util.repair as repair

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CEBareBonesFireworksAlgorithm']


class CEBareBonesFireworksAlgorithm(CoEvolutionOptimizationAlgorithm):
    r"""Implementation of Bare Bones Fireworks Algorithm for coevolution.

    Algorithm:
        Bare Bones Fireworks Algorithm

    Date:
        2023

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://www.sciencedirect.com/science/article/pii/S1568494617306609

    Reference paper:
        Junzhi Li, Ying Tan, The bare bones fireworks algorithm: A minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046.

    Attributes:
        Name (List[str]): List of strings representing algorithm names
        num_sparks (int): Number of sparks
        amplification_coefficient (float): amplification coefficient
        reduction_coefficient (float): reduction coefficient

    See Also:
        * :class:`niapy.algorithms.CoEvolutionOptimizationAlgorithm`

    """

    Name = ['CoEvolutionBareBonesFireworksAlgorithm', 'CE-BBFWA']

    @staticmethod
    def info():
        r"""Get default information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Junzhi Li, Ying Tan, The bare bones fireworks algorithm: A minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046."""

    def set_parameters(self, num_sparks=10, amplification_coefficient=1.5, reduction_coefficient=0.5, *args, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            num_sparks (int): Number of sparks :math:`\in [1, \infty)`.
            amplification_coefficient (float): Amplification coefficient :math:`\in [1, \infty)`.
            reduction_coefficient (float): Reduction coefficient :math:`\in (0, 1)`.

        """
        super().set_parameters(*args, **kwargs)
        self.num_sparks = num_sparks
        self.amplification_coefficient = amplification_coefficient
        self.reduction_coefficient = reduction_coefficient
        self.population_size = 1

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        params = super().get_parameters()
        params.update({
            'num_sparks': self.num_sparks,
            'amplification_coefficient': self.amplification_coefficient,
            'reduction_coefficient': self.reduction_coefficient
        })
        return params

    def init_population(self, task, groups):
        r"""Initialize starting population.

        Args:
            task (Task): Optimization task.
            groups (Union[list[int], set[int], numpy.ndarray])

        Returns:
            Tuple[numpy.ndarray, float, Dict[str, Any]]:
                1. Initial solution.
                2. Initial solution function/fitness value.
                3. Additional arguments:
                    * A (numpy.ndarray): Starting amplitude or search range.

        """
        x, x_fit, d = super().init_population(task)
        d.update({'amplitude': task.range})
        return x, x_fit, d

    def run_iteration(self, task, groups, population, population_fitness, best_x, best_fitness, *args, **params):
        r"""Core function of Bare Bones Fireworks Algorithm.

        Args:
            task (Task): Optimization task.
            groups (Union[list[int], set[int], numpy.ndarray])
            population (numpy.ndarray): Current solution.
            population_fitness (float): Current solution fitness/function value.
            best_x (numpy.ndarray): Current best solution.
            best_fitness (float): Current best solution fitness/function value.
            args (list): Additional parameters.
            params (dict[str, any]): Additional parameters.

        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray, float, Dict[str, Any]]:
                1. New solution.
                2. New solution fitness/function value.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * amplitude (numpy.ndarray): Search range.

        """
        amplitude = params.pop('amplitude')
        x, xf = population[0], population_fitness[0]
        sparks = np.tile(x, (self.num_sparks, 1))
        sparks[:, groups] = self.uniform(x[groups] - amplitude[groups], x[groups] + amplitude[groups], (self.num_sparks, len(groups)))
        sparks = np.apply_along_axis(task.repair, 1, sparks, self.rng)
        sparks_fitness = np.apply_along_axis(task.eval, 1, sparks)
        best_index = np.argmin(sparks_fitness)
        if sparks_fitness[best_index] < xf:
            x = sparks[best_index]
            xf = sparks_fitness[best_index]
            amplitude = self.amplification_coefficient * amplitude
        else:
            amplitude = self.reduction_coefficient * amplitude
        return [x], [xf], x.copy(), xf, {'amplitude': amplitude}

