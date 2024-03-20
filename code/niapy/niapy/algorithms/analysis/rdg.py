# encoding=utf8
import sys
import logging

from functools import reduce

import numpy as np

from niapy.algorithms.algorithm import (
    AnalysisAlgorithm,
    default_numpy_init
)

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.analysis')
logger.setLevel('INFO')

__all__ = [
    'RecursiveDifferentialGrouping',
    'RecursiveDifferentialGroupingV2',
    'RecursiveDifferentialGroupingV3',
    'EfficientRecursiveDifferentialGrouping',
    'ThreeLevelRecursiveDifferentialGrouping',
]


class RecursiveDifferentialGrouping(AnalysisAlgorithm):
    r"""Implementation of recursive differential grouping.

    Algorithm:
        Recursive Differential Grouping

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
        * :class:`niapy.algorithms.AnalysisAlgorithm`

    Attributes:
        alpha (float): Multiplier for epsilon.
        k (int): Numbner of solutions for determening the epsilon parameter.
    """
    Name = ['RecursiveDifferentialGrouping', 'RDG']

    def __init__(self, alpha=None, k=None, *args, **kwargs):
        r"""Initialize RecursiveDifferentialGrouping.

        Args:
            alpha (Optional[float]): Multiplier for epsilon.
            k (Optional[int]): Numbner of solutions for determening the epsilon parameter.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.
        """
        super().__init__(alpha=alpha, k=k, *args, **kwargs)

    def set_parameters(self, alpha=None, k=None, *args, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            alpha (Optional[float]): TODO.
            k (Optional[int]): Number of starting population.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.
        """
        super().set_parameters(**kwargs)
        self.alpha = alpha if alpha else 1e-12
        self.k = k if k else 10

    def get_parameters(self):
        r"""Get parameter values for the algorithm.

        Returns:
            dict[str, any]: Key-value.
        """
        d = super().get_parameters()
        d.update({
            'alpha': self.alpha,
            'k': self.k
        })
        return d

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`
        """
        return r"""Sun Y, Kirley M, Halgamuge S K. A Recursive Decomposition Method for Large Scale Continuous Optimization[J]. IEEE Transactions on Evolutionary Computation, 22, no. 5 (2018): 647-661."""

    def gamma(self, task, *args):
        r"""TODO

        Args:
            task (Task): Optimization task.
            *args (list[float]): Parameters for determening gamma.

        Returns:
            float: Value of gamma.
        """
        return args[0]

    def interact(self, task, a, af, epsilon, S1, S2, X_r):
        r"""Method for detecting interactions between componentes.

        Args:
            task (Task): Optimization task.
            a (numpy.ndarray): Solution with all components set to minium of optimization space.
            af (float): Fitness value of solution `a`.
            epsilon (float): Parameter for determening the interaction between components.
            S1 (list[int]): First set of components indexes to test for interaction.
            S2 (list[int]): Second set of components indexes to test for interaction.
            X_r (list[int]): Remaining components for determening the interactions.

        Returns:
            list[int]: TODO.
        """
        b, c, d = np.copy(a), np.copy(a), np.copy(a)
        b[S1] = d[S1] = task.upper[S1]
        bf = task.eval(b)
        d1 = af - bf
        c[S2] = d[S2] = task.lower[S2] + (task.upper[S2] - task.lower[S2]) / 2
        cf, df = task.eval(c), task.eval(d)
        d2 = cf - df
        S = list(S1)
        if np.abs(d1 - d2) > self.gamma(task, epsilon, af, bf, cf, df):
            if np.size(S2) == 1:
                S = np.union1d(S1, S2).tolist()
            else:
                k = int(np.floor(np.size(S2) / 2))
                S2_1, S2_2 = S2[:k], S2[k:]
                S1_1 = self.interact(task, a, af, epsilon, S1, S2_1, X_r)
                S1_2 = self.interact(task, a, af, epsilon, S1, S2_2, X_r)
                S = np.union1d(S1_1, S1_2).tolist()
        else:
            X_r.extend(S2)
        return S

    def run(self, task, *args, **kwargs):
        r"""Core function of RecursiveDifferentialGrouping algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[list[int]]:
        """
        _, fpop = default_numpy_init(task, self.k, self.rng)
        seps, allgroups = [], []
        epsilon = np.min(np.abs(fpop)) * self.alpha
        S1, S2 = [0], [i + 1 for i in range(task.dimension - 1)]
        X_r = [0]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while X_r:
            X_r = []
            S1_a = self.interact(task, p1, p1f, epsilon, S1, S2, X_r)
            if np.size(S1_a) == np.size(S1):
                if np.size(S1) == 1:
                    seps.extend(S1)
                else:
                    allgroups.append(S1)
                if np.size(X_r) > 1:
                    S1 = X_r[:1]
                    X_r = X_r[1:]
                    S2 = list(X_r)
                else:
                    seps.append(X_r[0])
                    break
            else:
                S1 = S1_a
                S2 = X_r
                if (np.size(X_r) == 0):
                    allgroups.append(S1)
                    break
        for e in seps: allgroups.append([e])
        return allgroups


class RecursiveDifferentialGroupingV2(RecursiveDifferentialGrouping):
    r"""Implementation of recursive differential grouping version 2.

    Algorithm:
        Recursive Differential Grouping V2

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
        * :class:`niapy.algorithms.AnalysisAlgorithm`
        * :class:`niapy.algorithms.analysis.RecursiveDifferentialGrouping`

    Attributes:
        alpha (float): Multiplier for epsilon.
    """
    Name = ['RecursiveDifferentialGroupingV2', 'RDGv2']

    def set_parameters(self, alpha=None, k=None, *args, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            alpha (Optional[float]): TODO.
            k (Optional[int]): Number of starting population.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.
        """
        super().set_parameters(k=0, **kwargs)
        self.alpha = alpha if alpha else 1e-12

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
        Recursive Differential Grouping V3

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
        * :class:`niapy.algorithms.AnalysisAlgorithm`
        * :class:`niapy.algorithms.analysis.RecursiveDifferentialGroupingV2`

    Attributes:
        eps_n (Optional[int]): Group size control parameter for non-separable components.
        eps_s (Optional[int]): Group size control parameter for separable componentes.
    """
    Name = ['RecursiveDifferentialGroupingV3', 'RDGv3']

    def __init__(self, eps_n=50, eps_s=50, *args, **kwargs):
        """Initialize RecursiveDifferentialGroupingV3.

        Args:
            eps_n (Optional[int]): Group size control parameter for non-separable components.
            eps_s (Optional[int]): Group size control parameter for separable components.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.
        """
        super().__init__(*args, **kwargs)

    def set_parameters(self, eps_n=None, eps_s=None, *args, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            eps_n (Optional[int]): Group size control parameter for non-separable components.
            eps_s (Optional[int]): Group size control parameter for separable components.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.
        """
        super().set_parameters(**kwargs)
        self.eps_n = eps_n if eps_n else 50
        self.eps_s = eps_s if eps_s else 50

    def get_parameters(self):
        r"""Get parameter values for the algorithm.

        Returns:
            dict[str, any]: Key-value.
        """
        d = super().get_parameters()
        d.pop('k', None)
        d.update({
            'eps_n': self.eps_n,
            'eps_s': self.eps_s
        })
        return d

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`
        """
        return r"""Sun Y, Li X, Erst A, Omidvar, M N. Decomposition for Large-scale Optimization Problems with Overlapping Components. In 2019 IEEE Congress on Evolutionary Computation (CEC), pp. 326-333. IEEE, 2019."""

    def run(self, task, *args, **kwargs):
        r"""Core function of RecursiveDifferentialGroupingV3 algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[list[int]]: Groups.
        """
        seps, allgroups = [], []
        S1, S2 = [0], [i + 1 for i in range(task.dimension - 1)]
        X_r = [0]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while X_r:
            X_r = []
            S1_a = self.interact(task, p1, p1f, 0, S1, S2, X_r)
            if np.size(S1_a) != np.size(S1) and np.size(S1_a) < self.eps_n:
                S1 = S1_a
                S2 = X_r
                if np.size(X_r) == 0:
                    allgroups.append(S1)
                    break
            else:
                if np.size(S1_a) == 1:
                    seps.extend(S1_a)
                else:
                    allgroups.append(S1_a)
                if np.size(X_r) > 1:
                    S1 = X_r[:1]
                    X_r = X_r[1:]
                    S2 = list(X_r)
                elif np.size(X_r) == 1:
                    seps.append(X_r[0])
                    break
        while len(seps) > self.eps_s:
            allgroups.append(seps[:self.eps_s])
            seps = seps[self.eps_s:]
        if seps: allgroups.append(seps)
        return allgroups


class EfficientRecursiveDifferentialGrouping(RecursiveDifferentialGroupingV2):
    r"""Implementation of efficient recursive differential grouping.

    Algorithm:
        Efficient Recursive Differential Grouping

    Date:
        2024

    Authors:
        Klemen Berkovic

    License:
        MIT

    Reference URL:
        https://ieeexplore.ieee.org/document/9141328/

    Reference paper:
        M. Yang, A. Zhou, C. Li and X. Yao, "An Efficient Recursive Differential Grouping for Large-Scale Continuous Problems," in IEEE Transactions on Evolutionary Computation, vol. 25, no. 1, pp. 159-171, Feb. 2021, doi: 10.1109/TEVC.2020.3009390. keywords: {Optimization;Computational efficiency;Geology;Computer science;Electronic mail;Computational complexity;Automation;Cooperative co-evolution (CC);decomposition;large-scale global optimization},

    See Also:
        * :class:`niapy.algorithms.Algorithm`
        * :class:`niapy.algorithms.AnalysisAlgorithm`
        * :class:`niapy.algorithms.analysis.RecursiveDifferentialGrouping`
        * :class:`niapy.algorithms.analysis.RecursiveDifferentialGroupingV2`
    """
    Name = ['EfficientRecursiveDifferentialGrouping', 'ERDG']

    def __init__(self, *args, **kwargs):
        """Initialize EfficientRecursiveDifferentialGrouping.

        Args:
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
        return r"""M. Yang, A. Zhou, C. Li and X. Yao, "An Efficient Recursive Differential Grouping for Large-Scale Continuous Problems," in IEEE Transactions on Evolutionary Computation, vol. 25, no. 1, pp. 159-171, Feb. 2021, doi: 10.1109/TEVC.2020.3009390. keywords: {Optimization;Computational efficiency;Geology;Computer science;Electronic mail;Computational complexity;Automation;Cooperative co-evolution (CC);decomposition;large-scale global optimization}"""

    def set_parameters(self, *args, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.
        """
        super().set_parameters(**kwargs)

    def get_parameters(self):
        r"""Get parameter values for the algorithm.

        Returns:
            dict[str, any]: Key-value.
        """
        d = super().get_parameters()
        d.pop('k', None)
        return d

    def interact(self, task, p1, p2, S1, S2, y):
        r"""Method for detecting interactions between componentes.

        Args:
            task (Task): Optimization task.
            p1 (numpy.ndarray): Solution with all components set to minium of optimization space.
            p2 (numpy.ndarray): Solution with some components set to minium and some set to maxium of optimization space.
            S1 (list[int]): First set of components indexes to test for interaction.
            S2 (list[int]): Second set of components indexes to test for interaction.
            y (list[int]): Fitness values for given points `p1` and `p2`.

        Returns:
            tuple[list[int], list[int]]:
                1. List of components indexes.
                2. Four funciton values.
        """
        non_sep, S = True, list(S1)
        if None in y:
            p3, p4 = np.copy(p1), np.copy(p2)
            p3[S2] = (task.upper[S2] + task.lower[S2]) / 2
            p4[S2] = (task.upper[S2] + task.lower[S2]) / 2
            p3f, p4f = task.eval(p3), task.eval(p4)
            y[2], y[3] = -p3f, p4f
            epsilon = self.gamma(task, 0, *y)
            if np.abs(np.sum(y)) <= epsilon:
                non_sep = False
        if non_sep:
            if np.size(S2) == 1:
                S = np.union1d(S1, S2).tolist()
            else:
                k = np.floor(np.size(S2) / 2).astype(int)
                S2_1, S2_2 = S2[:k], S2[k:]
                S1_1, yn = self.interact(task, p1, p2, S1, S2_1, [y[0], y[1], None, None])
                d = np.sum(y) - np.sum(yn)
                if d != 0:
                    if np.size(S1_1) == np.size(S1):
                        S1_2, _ = self.interact(task, p1, p2, S1, S2_2, y)
                    else:
                        S1_2, _ = self.interact(task, p1, p2, S1, S2_2, [y[0], y[1], None, None])
                    S = np.union1d(S1_1, S1_2).tolist()
                else:
                    S = S1_1
        return S, y

    def run(self, task, *args, **kwargs):
        r"""Core function of EfficientRecursiveDifferentialGrouping algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[list[int]]: Groups.
        """
        seps, allgroups = [], []
        S1, S2 = [0], [i + 1 for i in range(task.dimension - 1)]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while S2:
            p2 = np.copy(p1)
            p2[S1] = task.upper[S1]
            p2f = task.eval(p2)
            S1_a, _ = self.interact(task, p1, p2, S1, S2, [p1f, -p2f, None, None])
            if np.size(S1_a) == np.size(S1):
                if np.size(S1) == 1:
                    seps.append(S1[0])
                else:
                    allgroups.append(S1)
                S1 = S2[:1]
                S2 = S2[1:]
            else:
                S1 = S1_a
                S2 = [x for x in S2 if x not in S1]
            if not S2:
                if np.size(S1) > 1:
                    allgroups.append(S1)
                elif np.size(S1) == 1:
                    seps.append(S1[0])
        for e in seps:
            allgroups.append([e])
        return allgroups


class ThreeLevelRecursiveDifferentialGrouping(RecursiveDifferentialGrouping):
    r"""Implementation of three-level recursive differential grouping.

    Algorithm:
        Three Level Recursive Differential Grouping

    Date:
        2024

    Authors:
        Klemen Berkovic

    License:
        MIT

    Reference URL:
        https://ieeexplore.ieee.org/document/9154465

    Reference paper:
        H. -B. Xu, F. Li and H. Shen, "A Three-Level Recursive Differential Grouping Method for Large-Scale Continuous Optimization," in IEEE Access, vol. 8, pp. 141946-141957, 2020, doi: 10.1109/ACCESS.2020.3013661. keywords: {Optimization;Iron;Sociology;Covariance matrices;Linear programming;Power electronics;Large-scale continuous optimization;cooperative co-evolution (CC);differential grouping;trichotomy method},

    See Also:
        * :class:`niapy.algorithms.Algorithm`
        * :class:`niapy.algorithms.AnalysisAlgorithm`
        * :class:`niapy.algorithms.RecursiveDifferentialGrouping`

    Attributes:
        alpha (float): Multiplier for epsilon.
        k (int): Numbner of solutions for determening the epsilon parameter.
    """
    Name = ['ThreeLevelRecursiveDifferentialGrouping', 'TRDG']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`
        """
        return r"""H. -B. Xu, F. Li and H. Shen, "A Three-Level Recursive Differential Grouping Method for Large-Scale Continuous Optimization," in IEEE Access, vol. 8, pp. 141946-141957, 2020, doi: 10.1109/ACCESS.2020.3013661. keywords: {Optimization;Iron;Sociology;Covariance matrices;Linear programming;Power electronics;Large-scale continuous optimization;cooperative co-evolution (CC);differential grouping;trichotomy method}"""

    def interact(self, task, S1, S2, epsilon, p1, p2, d_1_2):
        r"""Method for detecting interactions between componentes.

        Args:
            task (Task): Optimization task.
            S1 (list[int]): Set of componets indexes.
            S2 (list[int]): Set of componets indexes.
            epsilon (float): Value for determening the interaction.
            p1 (numpy.nadarray): Solution with all components set to minium of optimization space.
            p2 (numpy.nadarray): TODO.
            d_1_2 (float): TODO.

        Returns:
            bool: TODO.
        """
        p3, p4 = np.copy(p1), np.copy(p2)
        p3[S2] = (task.upper[S2] + task.lower[S2]) / 2
        p4[S2] = (task.upper[S2] + task.lower[S2]) / 2
        p3f, p4f = task.eval(p3), task.eval(p4)
        if np.abs(d_1_2 - (p3f - p4f)) > epsilon:
            return True
        else:
            return False

    def group(self, task, S1, S2, epsilon, p1, p2, d_1_2):
        r"""Method for detecting interactions between componentes.

        Args:
            task (Task): Optimization task.
            S1 (list[int]): Set of componets indexes.
            S2 (list[int]): Set of componets indexes.
            epsilon (float): Value for determening the interaction.
            p1 (numpy.nadarray): Solution with all components set to minium of optimization space.
            p2 (numpy.nadarray): Solution with components set to minium of optimization space and some set to maximum of the search space.
            d_1_2 (float): TODO.

        Returns:
            list[int]: TODO.
        """
        S = list(S1)
        if self.interact(task, S1, S2, epsilon, p1, p2, d_1_2):
            if np.size(S2) == 1:
                S = np.union1d(S1, S2).tolist()
            elif np.size(S2) == 2:
                S2_1, S2_2 = S2[:1], S2[1:]
                S1_1 = self.group(task, S1, S2_1, epsilon, p1, p2, d_1_2)
                S1_2 = self.group(task, S1, S2_2, epsilon, p1, p2, d_1_2)
                S = np.union1d(S1_1, S1_2).tolist()
            else:
                k = np.floor(np.size(S2) / 3).astype(int)
                S2_1, S2_2, S2_3 = S2[:k], S2[k:k * 2], S2[k * 2:]
                S1_1 = self.group(task, S1, S2_1, epsilon, p1, p2, d_1_2)
                S1_2 = self.group(task, S1, S2_2, epsilon, p1, p2, d_1_2)
                S1_3 = self.group(task, S1, S2_3, epsilon, p1, p2, d_1_2)
                S = reduce(np.union1d, (S1_1, S1_2, S1_3)).tolist()
        return S

    def run(self, task, *args, **kwargs):
        r"""Core function of ThreeLevelRecursiveDifferentialGrouping algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[list[int]]:
        """
        _, fpop = default_numpy_init(task, np.floor(self.k / 4).astype(int), self.rng)
        epsilon = np.min(np.abs(fpop)) * self.alpha
        seps, allgroups = [], []
        S1, S2 = [0], [i + 1 for i in range(task.dimension - 1)]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while S2:
            p2 = np.copy(p1)
            p2[S1] = task.upper[S1]
            p2f = task.eval(p2)
            S1_a = self.group(task, S1, S2, epsilon, p1, p2, (p1f - p2f))
            if np.size(S1_a) == np.size(S1):
                if np.size(S1) == 1:
                    seps.append(S1[0])
                else:
                    allgroups.append(S1)
                S1 = S2[:1]
                S2 = S2[1:]
            else:
                S1 = S1_a
                S2 = [x for x in S2 if x not in S1]
            if not S2:
                if np.size(S1) > 1:
                    allgroups.append(S1)
                elif np.size(S1) == 1:
                    seps.append(S1[0])
        for e in seps: allgroups.append([e])
        return allgroups
