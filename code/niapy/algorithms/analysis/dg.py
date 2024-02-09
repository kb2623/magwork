# encoding=utf8
import sys
import logging

import numpy as np

from niapy.algorithms.algorithm import AnalysisAlgorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['DifferentialGrouping']


class DifferentialGrouping(AnalysisAlgorithm):
    r"""Implementation of differential grouping.

    Algorithm:
        DifferentialGrouping

    Date:
        2022

    Authors:
        Klemen Berkovic

    License:
        MIT

    Reference URL:
        https://ieeexplore.ieee.org/document/6595612

    Reference paper:
        M. N. Omidvar, X. Li, Y. Mei and X. Yao, "Cooperative Co-Evolution With Differential Grouping for Large Scale Optimization," in IEEE Transactions on Evolutionary Computation, vol. 18, no. 3, pp. 378-393, June 2014, doi: 10.1109/TEVC.2013.2281543.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Attributes:
        epsilon (float): TODO.

    """

    Name = ['DifferentialGrouping', 'DG']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""M. N. Omidvar, X. Li, Y. Mei and X. Yao, "Cooperative Co-Evolution With Differential Grouping for Large Scale Optimization," in IEEE Transactions on Evolutionary Computation, vol. 18, no. 3, pp. 378-393, June 2014, doi: 10.1109/TEVC.2013.2281543."""

    def __init__(self, epsilon=sys.float_info.epsilon, *args, **kwargs):
        """Initialize RecursiveDifferentialGrouping.

        Args:
            epsilon (Optional[float]): TODO.

        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def set_parameters(self, epsilon=None, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            epsilon (Optional[float]): TODO.

        """
        super().set_parameters(**kwargs)
        if epsilon: self.epsilon = epsilon

    def get_parameters(self):
        d = super().get_parameters()
        d.update({
            'epsilon': self.epsilon
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
        groups = []
        dims = [i for i in range(task.dimension)]
        for i in range(task.dimension):
            group = [i]
            for j in range(i + 1, task.dimension):
                p1, p2 = np.copy(task.lower), np.copy(task.lower)
                p2[i] = task.upper[i]
                d1 = task.eval(p1) - task.eval(p2)
                p1[j] = p2[j] = (task.upper[j] + task.lower[j]) / 2
                d2 = task.eval(p1) - task.eval(p2)
                if np.abs(d1 - d2) > self.epsilon: group.append(j)
            # ------------ END for ------------
            groups.append(group)
        # ------------ END while ------------
        return self.indirect_interaction_learning(task, groups)

