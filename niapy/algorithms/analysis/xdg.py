# encoding=utf8
import sys
import logging

import numpy as np

from niapy.algorithms.algorithm import (
    AnalysisAlgorithm,
    default_numpy_init
)

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['ExtendedDifferentialGrouping']


class ExtendedDifferentialGrouping(AnalysisAlgorithm):
    r"""Implementation of extended differential grouping.

    Algorithm:
        ExtendedDifferentialGrouping

    Date:
        2018

    Authors:
        Klemen Berkovic

    License:
        MIT

    Reference URL:
        https://dl.acm.org/doi/10.1145/2739480.2754666

    Reference paper:
        Yuan Sun, Michael Kirley, and Saman Kumara Halgamuge. 2015. Extended Differential Grouping for Large Scale Global Optimization with Direct and Indirect Variable Interactions. In Proceedings of the 2015 Annual Conference on Genetic and Evolutionary Computation (GECCO '15). Association for Computing Machinery, New York, NY, USA, 313–320. https://doi.org/10.1145/2739480.2754666

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Attributes:
        * epsilon (float): Change for searching in neighborhood.

    """

    Name = ['ExtendedDifferentialGrouping', 'XDG']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""Yuan Sun, Michael Kirley, and Saman Kumara Halgamuge. 2015. Extended Differential Grouping for Large Scale Global Optimization with Direct and Indirect Variable Interactions. In Proceedings of the 2015 Annual Conference on Genetic and Evolutionary Computation (GECCO '15). Association for Computing Machinery, New York, NY, USA, 313–320. https://doi.org/10.1145/2739480.2754666"""

    def __init__(self, epsilon=sys.float_info.epsilon, *args, **kwargs):
        """Initialize ExtendedDifferentialGrouping.

        Args:
            * epsilon (Optional[float]): Change for searching in neighborhood.

        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def set_parameters(self, epsilon=None, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            * epsilon (Optional[float]): Change for searching in neighborhood.
            * kwargs (dict): Additional keyword parameters.

        """
        super().set_parameters(**kwargs)
        if epsilon: self.epsilon = epsilon

    def get_parameters(self):
        d = super().get_parameters()
        d.update({
            'epsilon': self.epsilon,
        })
        return d

    def _count(self, item):
        r"""Get number of elements in a list.
    
        Args:
            item (Any): List with sublists

        Returns:
            int: Number of elements in a list
        """
        if isinstance(item, (list, tuple)): return np.sum(self._count(e) for e in item)
        else: return 1

    def direct_interaction_learning(self, task):
        r"""Direct interaction learning.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, List[List[int]]]:
                1. Interaction matrix
                2. Base groups
        """
        IM = np.zeros((task.dimension, task.dimension), dtype=float)
        groups = [[i] for i in range(task.dimension)]
        for i in range(task.dimension):
            x1, x2 = np.copy(task.lower), np.copy(task.lower)
            x2[i] = task.upper[i]
            d1 = task.eval(x1) - task.eval(x2)
            for j in range(i + 1, task.dimension):
                if IM[i, j] == 0:
                    x_1, x_2 = np.copy(x1), np.copy(x2)
                    x_1[j] = x_2[j] = (task.upper[j] + task.lower[j]) / 2
                    d2 = task.eval(x_1) - task.eval(x_2)
                    if np.abs(d1 - d2) > self.epsilon:
                        IM[i, j] = d1 - d2
                        groups[i].append(j)
                    # -- end if ------------------------------------------
                else:
                    groups[i].append(j)
                # -- end if ----------------------------------------------
            # -- end for -------------------------------------------------
            for p, q in [(k, l) for k in groups[i] for l in groups[i]]:
                if p >= q: break
                else: IM[p, q] = IM[q, p]
            # -- end for -------------------------------------------------
        # -- end for -----------------------------------------------------
        return IM, groups

    def indirect_interaction_learning(self, task, groups):
        r"""Indirect intreaction learning.

        Args:
            task (Task): Optimization task.
            groups (List[List[int]]): Base groups.

        Returns:
            List[List[int]]: Groups.
        """
        while self._count(groups) > task.dimension:
            p = 0
            while p < len(groups):
                q = p + 1
                while q < len(groups) and p < q:
                    if np.size(np.intersect1d(groups[p], groups[q])) != 0: 
                        groups[p] = np.union1d(groups[p], groups[q]).tolist()
                        del groups[q]
                    # -- end if ---------------------------------------------
                    q += 1
                # -- end while ----------------------------------------------
                p += 1
            # -- end while --------------------------------------------------
        # -- end while ------------------------------------------------------
        return groups

    def run(self, task, *args, **kwargs):
        r"""Core function of HillClimbAlgorithm algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[Union[list, list[int]]]: Groups.

        """
        # Direct Interaction Learning
        _, groups = self.direct_interaction_learning(task)        
        # Indirect Interaction Learning
        return self.indirect_interaction_learning(task, groups)

