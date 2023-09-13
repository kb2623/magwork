# encoding=utf8
import sys
import logging

import numpy as np

from niapy.algorithms.algorithm import AnalysisAlgorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['DualDifferentialGrouping']


class DualDifferentialGrouping(AnalysisAlgorithm):
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
        https://ieeexplore.ieee.org/document/9743365

    Reference paper:
        J. -Y. Li, Z. -H. Zhan, K. C. Tan and J. Zhang, "Dual Differential Grouping: A More General Decomposition Method for Large-Scale Optimization," in IEEE Transactions on Cybernetics, doi: 10.1109/TCYB.2022.3158391.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Attributes:
        epsilon_addit (float): TODO.
        epsilon_multi (float): TODO.

    """

    Name = ['DualDifferentialGrouping', 'DDG']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""J. -Y. Li, Z. -H. Zhan, K. C. Tan and J. Zhang, "Dual Differential Grouping: A More General Decomposition Method for Large-Scale Optimization," in IEEE Transactions on Cybernetics, doi: 10.1109/TCYB.2022.3158391."""

    def __init__(self, epsilon_addit=sys.float_info.epsilon, epsilon_multi=sys.float_info.epsilon, *args, **kwargs):
        """Initialize DualDifferentialGrouping.

        Args:
            epsilon_addit (Optional[float]): TODO.
            epsilon_multi (Optional[float]): TODO.

        """
        super().__init__(*args, **kwargs)
        self.epsilon_addit = epsilon_addit
        self.epsilon_multi = epsilon_multi

    def set_parameters(self, epsilon_addti=None, epsilon_multi=None, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            epsilon_addti (Optional[float]): TODO.
            epsilon_multi (Optional[float]): TODO.

        """
        kwargs.pop('population_size', None)
        super().set_parameters(**kwargs)
        if epsilon_addti: self.epsilon_addti = epsilon_addti
        if epsilon_multi: self.epsilon_multi = epsilon_multi

    def get_parameters(self):
        d = super().get_parameters()
        d.update({
            'epsilon_addit': self.epsilon_addit,
            'epsilon_multi': self.epsilon_multi
        })
        return d

    def _conv(self, item):
        r"""Get number of elements in a list.
    
        Args:
            item (Any): List with sublists

        Returns:
            int: Number of elements in a list
        """
        if isinstance(item, (list, tuple)): return [self._conv(e) for e in item]
        else: return int(item)


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
        r"""Core function of DualDifferentialGrouping algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[Union[list, list[int]]]: Groups.

        """
        seps, allgroups = [], []
        dims = [i for i in range(task.dimension)]
        x1 = np.copy(task.lower)
        y1 = task.eval(x1)
        while np.size(dims) > 0:
            i, tempgroup = dims[0], [dims[0]]
            del dims[0]
            x2 = np.copy(x1)
            x2[i] = task.upper[i]
            y2 = task.eval(x2)
            for j in dims:
                x3 = np.copy(x1)
                x3[j] = (task.upper[j] + task.lower[j]) / 2
                y3 = task.eval(x3)
                x4 = np.copy(x2)
                x4[j] = (task.upper[j] + task.lower[j]) / 2
                y4 = task.eval(x4)
                delta_addi, delta_multi = np.abs((y1 - y2) - (y3 - y4)), 0
                if y1 >= 0 and y2 >= 0 and y3 >= 0 and y4 >= 0: delta_multi = np.abs((np.log(y1) - np.log(y2)) - (np.log(y3) - np.log(y4)))
                else: delta_multi = self.epsilon_multi
                if delta_addi > self.epsilon_addit and delta_multi > self.epsilon_multi: tempgroup.append(j)
            if np.size(tempgroup) == 1: seps = np.union1d(seps, tempgroup)
            else: allgroups.append(tempgroup)
            # ---------- END if ----------
        # ---------- END while ----------
        for e in seps: allgroups.append([e])
        allgroups = self.indirect_interaction_learning(task, allgroups)
        return self._conv(allgroups)


def index_of(l, e):
    r"""Funtion

    Args:
        l (list): TODO.
        e (any): TODO.

    Returns:
        int: Index of element
    """
    for i, m in enumerate(l):
        if m == e: return i
    # ------------ END while ------------
    return -1

