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
    'ThreeLevelRecursiveDifferentialGrouping'
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
        * :class:`niapy.algorithms.AnalysisAlgorithm`

    Attributes:
        alpha (float): Multiplier for epsilon.
        n (int): Numbner of solutions for determening the epsilon parameter.
    """
    Name = ['RecursiveDifferentialGrouping', 'RDG']

    def __init__(self, alpha=None, n=None, *args, **kwargs):
        r"""Initialize RecursiveDifferentialGrouping.

        Args:
            alpha (Optional[float]): Multiplier for epsilon.
            n (Optional[int]): Numbner of solutions for determening the epsilon parameter.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.
        """
        super().__init__(alpha=alpha, n=n, *args, **kwargs)

    def set_parameters(self, alpha=None, n=None, *args, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            alpha (Optional[float]): TODO.
            n (Optional[int]): Number of starting population.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.
        """
        super().set_parameters(**kwargs)
        self.alpha = alpha if alpha else sys.float_info.epsilon
        self.n = n if n else 50

    def get_parameters(self):
        r"""Get parameter values for the algorithm.

        Returns:
            dict[str, any]: Key-value.
        """
        d = super().get_parameters()
        d.update({
            'alpha': self.alpha,
            'n': self.n
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

    def interact(self, task, a, af, epsilon, sub1, sub2, xremain):
        r"""Method for detecting interactions between componentes.
        
        Args:
            task (Task): Optimization task.
            a (numpy.ndarray): Solution with all components set to minium of optimization space.
            af (float): Fitness value of solution `a`.
            epsilon (float): TODO.
            sub1 (list[int]): TODO.
            sub2 (list[int]): TODO.
            xremain (list[int]): TODO.

        Returns:
            list[int]: TODO.
        """
        b, c, d = np.copy(a), np.copy(a), np.copy(a)
        b[sub1] = d[sub1] = task.upper[sub1]
        bf = task.eval(b)
        d1 = af - bf
        c[sub2] = d[sub2] = task.lower[sub2] + (task.upper[sub2] - task.lower[sub2]) / 2
        cf, df = task.eval(c), task.eval(d)
        d2 = cf - df
        sub1_n = list(sub1)
        if np.abs(d1 - d2) > self.gamma(task, epsilon, af, bf, cf, df):
            if np.size(sub2) == 1:
                sub1_n = np.union1d(sub1, sub2).tolist()
            else:
                k = int(np.floor(np.size(sub2) / 2))
                sub2_1 = [e for e in sub2[:k]]
                sub2_2 = [e for e in sub2[k:]]
                sub1_1 = self.interact(task, a, af, epsilon, sub1, sub2_1, xremain)
                sub1_2 = self.interact(task, a, af, epsilon, sub1, sub2_2, xremain)
                sub1_n = np.union1d(sub1_1, sub1_2).tolist()
        else:
            xremain.extend(sub2)
        return sub1_n

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
                if np.size(xremain) > 1:
                    sub1 = xremain[:1]
                    xremain = xremain[1:]
                    sub2 = xremain
                else:
                    seps.append(xremain[0])
                    break
            else:
                sub1 = sub1_a
                sub2 = xremain
                if (np.size(xremain) == 0):
                    allgroups.append(sub1)
                    break
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
        * :class:`niapy.algorithms.AnalysisAlgorithm`
        * :class:`niapy.algorithms.analysis.RecursiveDifferentialGrouping`
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
        d.pop('n', None)
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
        r"""Core function of HillClimbAlgorithm algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[Union[int, list[int]]]: Groups.
        """
        seps, allgroups = [], []
        sub1, sub2 = [0], [i + 1 for i in range(task.dimension - 1)]
        xremain = [0]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while len(xremain) > 0:
            xremain = []
            sub1_a = self.interact(task, p1, p1f, 0, sub1, sub2, xremain)
            if np.size(sub1_a) != np.size(sub1) and np.size(sub1_a) < self.eps_n:
                sub1 = sub1_a
                sub2 = xremain
                if np.size(xremain) == 0:
                    allgroups.append(sub1)
                    break
            else:
                if np.size(sub1_a) == 1:
                    seps.extend(sub1_a)
                else:
                    allgroups.append(sub1_a)
                if np.size(xremain) > 1:
                    sub1 = [xremain[0]]
                    del xremain[0]
                    sub2 = xremain
                elif np.size(xremain) == 1:
                    seps.append(xremain[0])
                    break
        while len(seps) > self.eps_s:
            allgroups.append(seps[:self.eps_s])
            seps = seps[self.eps_s:]
        if seps: allgroups.append(seps)
        return allgroups


class EfficientRecursiveDifferentialGrouping(RecursiveDifferentialGroupingV2):
    r"""Implementation of efficient recursive differential grouping.

    Algorithm:
        EfficientRecursiveDifferentialGrouping

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
        d.pop('n', None)
        return d

    def interact(self, task, p1, p2, sub1, sub2, y):
        r"""Method for detecting interactions between componentes.
        
        Args:
            task (Task): Optimization task.
            p1 (numpy.ndarray): Solution with all components set to minium of optimization space.
            p2 (numpy.ndarray): Solution with all components set to minium of optimization space.
            sub1 (list[int]): Set of componets indexes.
            sub2 (list[int]): Set of componets indexes.
            y (list[int]): TODO.

        Returns:
            tuple[list[int], list[int]]:
                1. List of components indexes.
                2. Four funciton values.
        """
        non_sep, sub1_n = True, list(sub1)
        if None in y:
            p3, p4 = np.copy(p1), np.copy(p2)
            p3[sub2] = (task.upper[sub2] + task.lower[sub2]) / 2 
            p4[sub2] = (task.upper[sub2] + task.lower[sub2]) / 2
            p3f, p4f = task.eval(p3), task.eval(p4)
            y[2], y[3] = -p3f, p4f
            epsilon = self.gamma(task, 0, *y)
            if np.abs(np.sum(y)) <= epsilon:
                non_sep = False
        if non_sep:
            if np.size(sub2) == 1:
                sub1_n = np.union1d(sub1, sub2).tolist()
            else:
                k = np.floor(np.size(sub2) / 2).astype(int)
                sub2_1, sub2_2 = sub2[:k], sub2[k:]
                sub1_1, yn = self.interact(task, p1, p2, sub1, sub2_1, [y[0], y[1], None, None])
                d = np.sum(y) - np.sum(yn)
                if d != 0:
                    if np.size(sub1_1) == np.size(sub1):
                        sub1_2, _ = self.interact(task, p1, p2, sub1, sub2_2, y)
                    else:
                        sub1_2, _ = self.interact(task, p1, p2, sub1, sub2_2, [y[0], y[1], None, None])
                    sub1_n = np.union1d(sub1_1, sub1_2).tolist()
                else:
                    sub1_n = sub1_1
        return sub1_n, y

    def run(self, task, *args, **kwargs):
        r"""Core function of HillClimbAlgorithm algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[Union[int, list[int]]]: Groups.
        """
        seps, allgroups = [], []
        sub1, sub2 = [0], [i + 1 for i in range(task.dimension - 1)]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while sub2:
            p2 = np.copy(p1)
            p2[sub1] = task.upper[sub1]
            p2f = task.eval(p2)
            sub1_a, _ = self.interact(task, p1, p2, sub1, sub2, [p1f, -p2f, None, None])
            if np.size(sub1_a) == np.size(sub1):
                if np.size(sub1) == 1:
                    seps.append(sub1[0])
                else:
                    allgroups.append(sub1)
                sub1 = [sub2[0]]
                sub2 = sub2[1:]
            else:
                sub1 = sub1_a
                sub2 = [x for x in sub2 if x not in sub1]
            if not sub2:
                if np.size(sub1) > 1:
                    allgroups.append(sub1)
                elif np.size(sub1) == 1:
                    seps.append(sub1[0])
        for e in seps:
            allgroups.append([e])
        return allgroups


class ThreeLevelRecursiveDifferentialGrouping(AnalysisAlgorithm):
    r"""Implementation of three-level recursive differential grouping.

    Algorithm:
        ThreeLevelRecursiveDifferentialGrouping

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

    Attributes:
        alpha (float): Multiplier for epsilon.
        k (int): Numbner of solutions for determening the epsilon parameter.
    """
    Name = ['ThreeLevelRecursiveDifferentialGrouping', 'TRDG']

    def __init__(self, alpha=None, k=None, *args, **kwargs):
        r"""Initialize RecursiveDifferentialGrouping.

        Args:
            alpha (Optional[float]): Multiplier for epsilon.
            n (Optional[int]): Numbner of solutions for determening the epsilon parameter.
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
        self.alpha = alpha if alpha else sys.float_info.epsilon
        self.k = k if k else 50

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
        return r"""H. -B. Xu, F. Li and H. Shen, "A Three-Level Recursive Differential Grouping Method for Large-Scale Continuous Optimization," in IEEE Access, vol. 8, pp. 141946-141957, 2020, doi: 10.1109/ACCESS.2020.3013661. keywords: {Optimization;Iron;Sociology;Covariance matrices;Linear programming;Power electronics;Large-scale continuous optimization;cooperative co-evolution (CC);differential grouping;trichotomy method}"""

    def interact(self, task, sub1, sub2, epsilon, p1, p2, d_1_2):
        r"""Method for detecting interactions between componentes.
        
        Args:
            task (Task): Optimization task.
            sub1 (list[int]): Set of componets indexes.
            sub2 (list[int]): Set of componets indexes.
            epsilon (float): Value for determening the interaction.
            p1 (numpy.nadarray): Solution with all components set to minium of optimization space.
            p2 (numpy.nadarray): TODO.
            d_1_2 (float): TODO.

        Returns:
            bool: TODO.
        """
        p3, p4 = np.copy(p1), np.copy(p2)
        p3[sub2] = (task.upper[sub2] + task.lower[sub2]) / 2 
        p4[sub2] = (task.upper[sub2] + task.lower[sub2]) / 2
        p3f, p4f = task.eval(p3), task.eval(p4)
        if np.abs(d_1_2 - (p3f - p4f)) > epsilon:
            return True
        else:
            return False

    def group(self, task, sub1, sub2, epsilon, p1, p2, d_1_2):
        r"""Method for detecting interactions between componentes.
        
        Args:
            task (Task): Optimization task.
            sub1 (list[int]): Set of componets indexes.
            sub2 (list[int]): Set of componets indexes.
            epsilon (float): Value for determening the interaction.
            p1 (numpy.nadarray): Solution with all components set to minium of optimization space.
            p2 (numpy.nadarray): Solution with components set to minium of optimization space and some set to maximum of the search space.
            d_1_2 (float): TODO.

        Returns:
            bool: TODO.
        """
        sub1_n = list(sub1)
        if self.interact(task, sub1, sub2, epsilon, p1, p2, d_1_2):
            if np.size(sub2) == 1:
                sub1_n = np.union1d(sub1, sub2).tolist()
            elif np.size(sub2) == 2:
                sub2_1, sub2_2 = sub2[:1], sub2[1:]
                sub1_1 = self.group(task, sub1, sub2_1, epsilon, p1, p2, d_1_2)
                sub1_2 = self.group(task, sub1, sub2_2, epsilon, p1, p2, d_1_2)
                sub1_n = np.union1d(sub1_1, sub1_2).tolist()
            else:
                k = np.floor(np.size(sub2) / 3).astype(int)
                sub2_1, sub2_2, sub2_3 = sub2[:k], sub2[k:k * 2], sub2[k * 2:]
                sub1_1 = self.group(task, sub1, sub2_1, epsilon, p1, p2, d_1_2)
                sub1_2 = self.group(task, sub1, sub2_2, epsilon, p1, p2, d_1_2)
                sub1_3 = self.group(task, sub1, sub2_3, epsilon, p1, p2, d_1_2)
                sub1_n = reduce(np.union1d, (sub1_1, sub1_2, sub1_3)).tolist()
        return sub1_n

    def run(self, task, *args, **kwargs):
        r"""Core function of RecursiveDifferentialGrouping algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[Union[list, list[int]]]:
        """
        _, fpop = default_numpy_init(task, np.floor(self.k / 4).astype(int), self.rng)
        epsilon = np.min(np.abs(fpop)) * self.alpha
        seps, allgroups = [], []
        sub1, sub2 = [0], [i + 1 for i in range(task.dimension - 1)]
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        while sub2:
            p2 = np.copy(p1)
            p2[sub1] = task.upper[sub1]
            p2f = task.eval(p2)
            sub1_a = self.group(task, sub1, sub2, epsilon, p1, p2, (p1f - p2f))
            if np.size(sub1_a) == np.size(sub1):
                if np.size(sub1) == 1:
                    seps.append(sub1[0])
                else:
                    allgroups.append(sub1)
                sub1 = [sub2[0]]
                sub2 = sub2[1:]
            else:
                sub1 = sub1_a
                sub2 = [x for x in sub2 if x not in sub1]
            if not sub2:
                if np.size(sub1) > 1:
                    allgroups.append(sub1)
                elif np.size(sub1) == 1:
                    seps.append(sub1[0])
        for e in seps: allgroups.append([e])
        return allgroups
