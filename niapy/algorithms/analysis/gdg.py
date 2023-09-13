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

__all__ = ['GlobalDifferentialGrouping']


class GlobalDifferentialGrouping(AnalysisAlgorithm):
    r"""Implementation of global differential grouping.

    Algorithm:
        DifferentialGrouping

    Date:
        2022

    Authors:
        Klemen Berkovic

    License:
        MIT

    Reference URL:
        https://dl.acm.org/doi/10.1145/2791291

    Reference paper:
        Yi Mei, Mohammad Nabi Omidvar, Xiaodong Li, and Xin Yao. 2016. A Competitive Divide-and-Conquer Algorithm for Unconstrained Large-Scale Black-Box Optimization. ACM Trans. Math. Softw. 42, 2, Article 13 (June 2016), 24 pages. https://doi.org/10.1145/2791291

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Attributes:
        * epsilon (float): TODO.
        * n (int): Number of trial vectors to calculate alpha.

    """

    Name = ['GlobalDifferentialGrouping', 'GDG']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""Yi Mei, Mohammad Nabi Omidvar, Xiaodong Li, and Xin Yao. 2016. A Competitive Divide-and-Conquer Algorithm for Unconstrained Large-Scale Black-Box Optimization. ACM Trans. Math. Softw. 42, 2, Article 13 (June 2016), 24 pages. https://doi.org/10.1145/2791291"""

    def __init__(self, epsilon=sys.float_info.epsilon, n=50, *args, **kwargs):
        """Initialize GlobalDifferentialGrouping.

        Args:
            epsilon (Optional[float]): TODO.
            n (Optional[int]): Number of individualt to evalute to calculate alpha.
            args (list): Additional list arguments.
            kwargs (dict): Additional keywoard arguments.

        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.n = n

    def set_parameters(self, epsilon=None, n=None, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            epsilon (Optional[float]): TODO.
            n (int): Number of individualt to evalute to calculate alpha.
            kwargs (dict): Additional keyword arguments.

        """
        super().set_parameters(**kwargs)
        if epsilon: self.epsilon = epsilon
        if n: self.n = n

    def get_parameters(self):
        d = super().get_parameters()
        d.update({
            'epsilon': self.epsilon,
            'n': self.n
        })
        return d

    def graph_connected_components(self, task, C, trash):
        r"""Calculate treshold from the delata matrix.

        Args:
            task (Task): Optimization task.
            C (numpy.ndarray): Connection matrix.
            trash (float): TODO.

        Returns:
            Tuple[List[int], List[int]]:
                1. TODO.
                2. TODO.
        """
        # Breadth-first search:
        labels = [0 for i in range(task.dimension)] # all vertex unexplored at the begining
        rts = []
        ccc = 0 # connected components counter
        while True:
            ind = find_all(labels, 0)
            if np.size(ind) != 0:
                fue = ind[0] # first unexplored vertex
                rts.append(fue)
                mlist = [fue]
                ccc += 1
                labels[fue] = ccc
                while True:
                    mlist_new = []
                    for lc in range(np.size(mlist)):
                        cp = find_tresh(C[mlist[lc]], trash) # points connected to point lc
                        for e in cp:
                            # get only unexplored vertecies
                            if labels[e] == 0:
                                mlist_new.append(e)
                                labels[e] = ccc
                            # ----------- END if -----------
                        # ----------- END for -----------
                    # ----------- END for -----------
                    mlist = mlist_new
                    if np.size(mlist) == 0:
                        break
                    # ----------- END if -----------
                # ----------- END while -----------
            else:
                break
            # ----------- END if -----------
        # ----------- END while -----------
        return labels, rts

    def run(self, task, *args, **kwargs):
        r"""Core function of GlobalDifferentialGrouping algorithm.

        Args:
            task (Task): Optimization task.
            args (list): Additional list parameters.
            kwargs (dict): Additional keyword parametes.

        Returns:
            list[Union[list, list[int]]]: Groups.

        """
        # Randomly sample 10 points
        _, fpop = default_numpy_init(task, self.n, self.rng)
        tresh = np.min(np.abs(fpop)) * self.epsilon
        # Algorithm start
        p1 = np.copy(task.lower)
        p1f = task.eval(p1)
        p2_vec, p3_vec = [], []
        for i in range(task.dimension):
            p2 = np.copy(p1)
            p2[i] = task.upper[i]
            p2_vec.append(task.eval(p2))
        # ----------- END for -----------
        for i in range(task.dimension):
            p3 = np.copy(p1)
            p3[i] = (task.upper[i] + task.lower[i]) / 2
            p3_vec.append(task.eval(p3))
        # ----------- END for -----------
        deltaMtx = np.zeros([task.dimension, task.dimension], dtype=float)
        for i, j in [(k, l) for k in range(task.dimension) for l in range(task.dimension)]:
            p4 = np.copy(p1)
            p4[i] = task.upper[i]
            p4[j] = (task.upper[j] + task.lower[j]) / 2
            p4f = task.eval(p4)
            d1, d2 = p1f - p2_vec[i], p3_vec[j] - p4f
            deltaMtx[i, j] = deltaMtx[j, i] = np.abs(d1 - d2)
        # ----------- END for -----------
        # group the variables according to the matrix
        labels, rst = self.graph_connected_components(task, deltaMtx, tresh)
        # transform labels to group_idx
        group_idx = np.copy(labels)
        for i in range(-1, np.max(labels) + 1):
            if np.count_nonzero(labels == i) == 1:
                for j in range(np.size(labels)):
                    if labels[j] == i: group_idx[labels[j]] = -1
                    elif labels[j] > i: group_idx[labels[j]] -= 1
                    # ----------- END if -----------
                # ----------- END for -----------
            # ----------- END if -----------
        # ----------- END for -----------
        seps, allgroups = find_all(group_idx, -1), []
        for i in range(np.max(group_idx) + 1):
            g = find_all(group_idx, i)
            if np.size(g) != 0: allgroups.append(g)
        # ----------- END for -----------
        for e in seps: allgroups.append([e])
        return allgroups


def find_all(arr, ele):
    r"""Find all elements.

    Args:
        arr (list): Input list for search.
        ele (Any): Element to search for.

    Returns:
        List[int]: List of indexses of a searched element.
    """
    ret = []
    for i, e in enumerate(arr):
        if e == ele: ret.append(i)
    # ----------- END for -----------
    return ret


def find_tresh(arr, tresh):
    r"""Find all elements higher than tresh.

    Args:
        arr (list): Input list for search.
        tresh (Any): Element to search for.

    Returns:
        List[int]: List of indexses of a searched element.
    """
    r = []
    for i, e in enumerate(arr):
        if e > tresh: r.append(i)
    # ----------- END for -----------
    return r
