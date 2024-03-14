# encoding=utf8
import sys
from typing import Self, Tuple
import logging

import numpy as np

from niapy.task import Task
from niapy.algorithms.algorithm import AnalysisAlgorithm
from niapy.algorithms.algorithm import OptimizationAlgorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CooperativeCoevolution']


class CCTask(Task):
    def __init__(self, task:Task, inds:np.ndarray, *argrs, **kwargs):
        self.task = task
        self.inds = inds
        self.dimension = len(self.inds)
        self.lower = self.task.lower[self.inds]
        self.upper = self.task.upper[self.inds]
        self.range = self.task.range[self.inds]
        self.repair_function = self.task.repair_function
        self.max_iters = 10000000
            
    def stopping_condition(self:Self) -> bool:
        return self.task.stopping_condition()

    def eval(self:Self, x:np.ndarray) -> float:
        xn = np.copy(self.task.lower)
        xn[self.inds] = x
        return self.task.eval(xn)


class CooperativeCoevolution(OptimizationAlgorithm):

    Name:list[str]=['CooperativeCoevolution', 'CC']

    def __init__(self:Self, decompozer:AnalysisAlgorithm, toptimizer:type[OptimizationAlgorithm], *args:list, **kwargs:dict[str, any]) -> None:
        super().__init__(decompozer=decompozer, toptimizer=toptimizer, *args, **kwargs)

    def set_parameters(self:Self, decompozer:AnalysisAlgorithm, toptimizer:type[OptimizationAlgorithm], population_size=65, **kwargs:dict[str, any]) -> None:
        super().set_parameters(**kwargs)
        self.population_size = population_size
        # set decompozer
        self.decompozer = decompozer
        # set optimizer
        self.toptimizer = toptimizer
        # set empty optimizer parameters
        self.toptimizer_params = {}

    def get_parameters(self:Self) -> dict[str, any]:
        params = super().get_parameters()
        params.update({
            'population_size': self.population_size,
            'decompozer': self.decompozer.Name[0],
            'optimizer': self.toptimizer.Name[0]
        })
        return params

    def set_decomposer_parameters(self:Self, *args:list, **kwargs:dict[str, any]) -> None:
        self.decompozer.set_parameters(*args, **kwargs)

    def get_decomposer_parameters(self:Self, *args:list, **kwargs:dict[str, any]) -> dict[str, any]:
        return self.decompozer.get_parameters()

    def set_optimizer_parameters(self:Self, *args:list, **kwargs:dict[str, any]) -> None:
        self.toptimizer_params = kwargs

    def get_optimizer_parameters(self:Self, *args:list, **kwargs:dict[str, any]) -> dict[str, any]:
        return self.toptimizer_params

    def _get_pop_size(self:Self, no_elems:int) -> int:
        r = no_elems * 10
        return r if r < self.population_size else self.population_size

    def init_population(self:Self, task:Task) -> Tuple[np.ndarray, np.ndarray, dict[str, any]]:
        # get groups
        groups, seps = [], []
        for e in self.decompozer.run(task):
            if len(e) == 1: seps.append(e[0])
            else: groups.append(e)
        # init task for algorithms
        tasks = [CCTask(task, g) for g in groups]
        if len(seps) > 0: tasks.append(CCTask(task, seps))
        # init algorithms based on group sizes 
        pop, popf = [], []
        algs, algs_params = [], []
        best_x = []
        for t in tasks:
            no_pop = self._get_pop_size(t.dimension)
            a = self.toptimizer(population_size=no_pop, seed=self.integers(sys.maxsize))
            a.set_parameters(**self.toptimizer_params)
            p, pf, d = a.init_population(t)
            algs.append(a)
            algs_params.append(d)
            pop.append(p)
            popf.append(pf)
        return pop, popf, {'groups': groups, 'seps': seps, 'tasks': tasks, 'algs': algs, 'algs_params': algs_params}

    def _get_x_best(self:Self, tasks:list[CCTask], xbs:list[np.ndarray]):
        x = np.copy(tasks[0].task.lower)
        for i, e in enumerate(xbs):
            x[tasks[i].inds] = e
        return x

    def get_best(self:Self, pop:list[np.ndarray], pop_f:list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        xbs, xbs_f = [], []
        for i, fs in enumerate(pop_f):
            xb_i = np.argmin(fs)
            xbs.append(np.copy(pop[i][xb_i]))
            xbs_f.append(fs[xb_i])
        return xbs, xbs_f

    def iteration_generator(self, task):
        iters = 0
        pop, fpop, params = self.init_population(task)
        xbs, fxbs = self.get_best(pop, fpop)
        xb = self._get_x_best(params['tasks'], xbs)
        fxb = task.eval(xb)
        if task.stopping_condition(): yield xb, fxb
        while True:
            pop, fpop, xb, fxb, xbs, fxbs, params = self.run_iteration(task, pop, fpop, xb, fxb, iters, xbs, fxbs, **params)
            iters += 1
            yield xb, fxb

    def run_iteration(self:Self, task:Task, pop:np.ndarray, fpop:np.ndarray, xb:np.ndarray, fxb:np.ndarray, iters:int, xbs:list[np.ndarray], fxbs:list[float], tasks:list[Task], algs:list[OptimizationAlgorithm], algs_params:list[dict[str, any]], groups:list[list[int]], seps:list[int], *args, **params:dict[str, any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, any]]:
        update = False
        for i, t in enumerate(tasks):
            oxb = np.copy(xbs[i])
            pop[i], fpop[i], xbs[i], fxbs[i], algs_params[i] = algs[i].run_iteration(t, pop[i], fpop[i], xbs[i], fxbs[i], iters=iters, **algs_params[i])
            if np.sum(oxb - xbs[i]) > 0: update = True
            if fxbs[i] < fxb:
                xb = np.copy(task.lower)
                xb[t.inds] = xbs[i]
                fxb = fxbs[i]
        if update:
            x = self._get_x_best(tasks, xbs)
            fx = task.eval(x)
            if fx < fxb:
                fxb = fx
                xb = np.copy(x)
        return pop, fpop, xb, fxb, xbs, fxbs, {'groups': groups, 'seps': seps, 'tasks': tasks, 'algs': algs, 'algs_params': algs_params}

