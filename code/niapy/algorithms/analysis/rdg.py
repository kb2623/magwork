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

__all__ = [
    'RecursiveDifferentialGrouping',
    'RecursiveDifferentialGroupingV2',
    'RecursiveDifferentialGroupingV3'
]


class RecursiveDifferentialGrouping(AnalysisAlgorithm):
    r"""Implementation of recursive differential grouping.

    Algorithm:
        RecursiveDifferentialGrouping

    Date:
        2022

    Authors:
        Klemen Berkovic

    License:
        MIT

    Reference URL:
        https://ieeexplore.ieee.org/abstract/document/8122017

    Reference paper:
        Sun Y, Kirley M, Halgamuge S K. A Recursive Decomposition Method for Large Scale Continuous Optimization[J]. IEEE Transactions on Evolutionary Computation, 22, no. 5 (2018): 647-661.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Attributes:
        alpha (float): TODO.
        n (int): Numbner of starting population.

    """

    Name = ['RecursiveDifferentialGrouping', 'RDG']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""Sun Y, Kirley M, Halgamuge S K. A Recursive Decomposition Method for Large Scale Continuous Optimization[J]. IEEE Transactions on Evolutionary Computation, 22, no. 5 (2018): 647-661."""

    def __init__(self, alpha=sys.float_info.epsilon, n=50, *args, **kwargs):
        """Initialize RecursiveDifferentialGrouping.

        Args:
            alpha (Optional[float]): TODO.
            n (Optional[int]): Number of starting population.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.n = n

    def set_parameters(self, alpha=None, n=None, *args, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            alpha (Optional[float]): TODO.
            n (int): Number of starting population.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        """
        super().set_parameters(**kwargs)
        if alpha: self.alpha = alpha
        if n: self.n = n

    def get_parameters(self):
        d = super().get_parameters()
        d.update({
            'alpha': self.alpha,
            'n': self.n
        })
        return d

    def gamma(self, task, *args):
        r"""TODO

        Args:
            task (Task): Optimization task.
            *args (float): TODO.

        Returns:
            float: Value of gamma.
        """
        return args[0]

    def interact(self, task, a, af, epsilon, sub1, sub2, xremain):
        r"""TODO
        
        Args:
            task (Task): Optimization task.
            a (numpy.ndarray): TODO.
            af (float): TODO.
            epsilon (float): TODO.
            sub1 (numpy.ndarray): TODO.
            sub2 (numpy.ndarray): TODO.
            xremain (numpy.ndarray): TODO.

        Returns:
            numpy.ndarray: TODO.
        """
        b, c, d = np.copy(a), np.copy(a), np.copy(a)
        b[sub1] = d[sub1] = task.upper[sub1]
        bf = task.eval(b)
        d1 = af - bf
        c[sub2] = d[sub2] = task.lower[sub2] + (task.upper[sub2] - task.lower[sub2]) / 2
        cf, df = task.eval(c), task.eval(d)
        d2 = cf - df
        if np.abs(d1 - d2) > self.gamma(task, epsilon, af, bf, cf, df):
            if np.size(sub2) == 1:
                sub1 = np.union1d(sub1, sub2).tolist()
            else:
                k = int(np.floor(np.size(sub2) / 2))
                sub2_1 = [e for e in sub2[:k]]
                sub2_2 = [e for e in sub2[k:]]
                sub1_1 = self.interact(task, a, af, epsilon, sub1, sub2_1, xremain)
                sub1_2 = self.interact(task, a, af, epsilon, sub1, sub2_2, xremain)
                sub1 = np.union1d(sub1_1, sub1_2).tolist()
            # ---------- END if ----------
        else:
            xremain.extend(sub2)
        # ---------- END if ----------
        return sub1

    def run(self, task, *args, **kwargs):
        r"""Core function of RecursiveDifferentialGrouping algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[Union[list, list[int]]]:

        """
        _, fpop = default_numpy_init(task, self.n, self.rng)
        seps, allgroups = [], []
        epsilon = np.min(np.abs(fpop)) * self.alpha
        sub1, sub2 = [0], [i + 1 for i in range(task.dimension - 1)]
        xremain = [0]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while len(xremain) > 0:
            xremain = []
            sub1_a = self.interact(task, p1, p1f, epsilon, sub1, sub2, xremain)
            if np.size(sub1_a) == np.size(sub1):
                if np.size(sub1) == 1:
                    seps.extend(sub1)
                else:
                    allgroups.append(sub1)
                # ---------- END if ----------
                if np.size(xremain) > 1:
                    sub1 = [xremain[0]]
                    del xremain[0]
                    sub2 = xremain
                else:
                    seps.append(xremain[0])
                    break
                # ---------- END if ----------
            else:
                sub1 = sub1_a
                sub2 = xremain
                if (np.size(xremain) == 0):
                    allgroups.append(sub1)
                    break
                # ---------- END if ----------
            # ---------- END if ----------
        # ---------- END while ----------
        for e in seps: allgroups.append([e])
        return allgroups


class RecursiveDifferentialGroupingV2(RecursiveDifferentialGrouping):
    r"""Implementation of recursive differential grouping version 2.

    Algorithm:
        RecursiveDifferentialGroupingV2

    Date:
        2022

    Authors:
        Klemen Berkovic

    License:
        MIT

    Reference URL:
        https://research.monash.edu/en/publications/adaptive-threshold-parameter-estimation-with-recursive-differenti

    Reference paper:
        Sun Y, Omidvar, M N, Kirley M, Li X. Adaptive Threshold Parameter Estimation with Recursive Differential Grouping for Problem Decomposition. In Proceedings of the Genetic and Evolutionary Computation Conference, pp. 889-896. ACM, 2018.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Attributes:
        alpha (float): TODO.

    """

    Name = ['RecursiveDifferentialGroupingV2', 'RDGv2']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""Sun Y, Omidvar, M N, Kirley M, Li X. Adaptive Threshold Parameter Estimation with Recursive Differential Grouping for Problem Decomposition. In Proceedings of the Genetic and Evolutionary Computation Conference, pp. 889-896. ACM, 2018."""

    def gamma(self, task, *args):
        r"""TODO

        Args:
            task (Task): Optimization task.
            *args (float): TODO.

        Returns:
            float: Value of gamma.
        """
        n = np.sum(np.abs(args[1:])) * (np.power(task.dimension, 0.5) + 2)
        mu = n * (self.alpha / 2)
        return mu / (1 - mu)


class RecursiveDifferentialGroupingV3(RecursiveDifferentialGroupingV2):
    r"""Implementation of recursive differential grouping version 3.

    Algorithm:
        RecursiveDifferentialGroupingV3

    Date:
        2022

    Authors:
        Klemen Berkovic

    License:
        MIT

    Reference URL:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8790204&tag=1

    Reference paper:
        Sun Y, Li X, Erst A, Omidvar, M N. Decomposition for Large-scale Optimization Problems with Overlapping Components. In 2019 IEEE Congress on Evolutionary Computation (CEC), pp. 326-333. IEEE, 2019.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Attributes:
        alpha (float): TODO.
        tn (int): TODO.

    """

    Name = ['RecursiveDifferentialGroupingV3', 'RDGv3']

    def __init__(self, tn=50, *args, **kwargs):
        """Initialize RecursiveDifferentialGrouping.

        Args:
            alpha (Optional[float]): TODO.
            n (Optional[int]): Number of starting population.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""Sun Y, Li X, Erst A, Omidvar, M N. Decomposition for Large-scale Optimization Problems with Overlapping Components. In 2019 IEEE Congress on Evolutionary Computation (CEC), pp. 326-333. IEEE, 2019."""

    def set_parameters(self, tn=None, *args, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            n (Optional[int]): TODO.
            alpha (Optional[float]): TODO.
            tn (Optional[int]): Number of starting population.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        """
        super().set_parameters(**kwargs)
        self.tn = tn if tn else 50

    def get_parameters(self):
        d = super().get_parameters()
        d.pop('n', None)
        d.update({
            'tn': self.tn
        })
        return d

    def run(self, task, *args, **kwargs):
        r"""Core function of HillClimbAlgorithm algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[Union[list, list[int]]]: Groups.

        """
        seps, allgroups = [], []
        sub1, sub2 = [0], [i + 1 for i in range(task.dimension - 1)]
        xremain = [0]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while len(xremain) > 0:
            xremain = []
            sub1_a = self.interact(task, p1, p1f, 0, sub1, sub2, xremain)
            if np.size(sub1_a) != np.size(sub1) and np.size(sub1_a) < self.tn:
                sub1 = sub1_a
                sub2 = xremain
                if np.size(xremain) == 0:
                    allgroups.append(sub1)
                    break
                # ---------- END if ----------
            else:
                if np.size(sub1_a) == 1: seps.extend(sub1_a)
                else: allgroups.append(sub1_a)
                if np.size(xremain) > 1:
                    sub1 = [xremain[0]]
                    del xremain[0]
                    sub2 = xremain
                elif np.size(xremain) == 1:
                    seps.append(xremain[0])
                    break
                # ---------- END if ----------
            # ---------- END if ----------
        # ---------- END while ----------
        for e in seps: allgroups.append([e])
        return allgroups

