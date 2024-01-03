# encoding=utf8
import logging
import math

import numpy as np

from niapy.algorithms.algorithm import OptimizationAlgorithm, Individual, default_individual_init
from niapy.util.array import objects_to_array

__all__ = ['DifferentialEvolution', 'DynNpDifferentialEvolution',
           'cross_rand1', 'cross_rand2', 'cross_best2', 'cross_best1', 'cross_best2', 'cross_curr2rand1',
           'cross_curr2best1', 'multi_mutations']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


def cross_rand1(pop, ic, f, cr, rng, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation strategy uses three different random individuals from population to perform mutation.

    Mutation:
        Name: DE/rand/1

        :math:`\mathbf{x}_{r_1, G} + differential_weight \cdot (\mathbf{x}_{r_2, G} - \mathbf{x}_{r_3, G}`
        where :math:`r_1, r_2, r_3` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: Mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    r1, r2, r3 = ic, ic, ic
    while r1 != ic: r1 = rng.integers(len(pop))
    while r2 != ic and r2 != r1: r2 = rng.integers(len(pop))
    while r3 != ic and r3 != r2 and r3 != r1: r3 = rng.integers(len(pop))
    x = [pop[r1][i] + f * (pop[r2][i] - pop[r3][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_best1(pop, ic, f, cr, rng, x_b=None, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation strategy uses two different random individuals from population and global best individual.

    Mutation:
        Name: de/best/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{best, G} + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G})`
        where :math:`r_1, r_2` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        x_b (Individual): Current global best individual.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    returns:
        numpy.ndarray: Mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    r1, r2 = ic, ic
    while r1 != ic: r1 = rng.integers(len(pop))
    while r2 != ic and r2 != r1: r2 = rng.integers(len(pop))
    x = [x_b[i] + f * (pop[r1][i] - pop[r2][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_rand2(pop, ic, f, cr, rng, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation strategy uses five different random individuals from population.

    Mutation:
        Name: de/best/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{r_1, G} + differential_weight \cdot (\mathbf{x}_{r_2, G} - \mathbf{x}_{r_3, G}) + differential_weight \cdot (\mathbf{x}_{r_4, G} - \mathbf{x}_{r_5, G})`
        where :math:`r_1, r_2, r_3, r_4, r_5` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    r1, r2, r3, r4, r5 = ic, ic, ic, ic, ic
    while r1 != ic: r1 = rng.integers(len(pop))
    while r2 != ic and r2 != r1: r2 = rng.integers(len(pop))
    while r3 != ic and r3 != r2 and r3 != r1: r3 = rng.integers(len(pop))
    while r4 != ic and r4 != r1 and r4 != r2 and r4 != r3: r4 = rng.integers(len(pop))
    while r5 != ic and r5 != r1 and r5 != r2 and r5 != r3 and r5 != r4: r5 = rng.integers(len(pop))
    x = [pop[r1][i] + f * (pop[r2][i] - pop[r3][i]) + f * (
                pop[r4][i] - pop[r5][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_best2(pop, ic, f, cr, rng, x_b=None, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation:
        Name: de/best/2

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{best, G} + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + differential_weight \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
        where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        x_b (Individual): Current global best individual.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    r1, r2, r3, r4 = ic, ic, ic, ic
    while r1 != ic: r1 = rng.integers(len(pop))
    while r2 != ic and r2 != r1: r2 = rng.integers(len(pop))
    while r3 != ic and r3 != r2 and r3 != r1: r3 = rng.integers(len(pop))
    while r4 != ic and r4 != r1 and r4 != r2 and r4 != r3: r4 = rng.integers(len(pop))
    x = [x_b[i] + f * (pop[r1][i] - pop[r2][i]) + f * (
                pop[r3][i] - pop[r4][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_curr2rand1(pop, ic, f, cr, rng, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation:
        Name: de/curr2rand/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{i, G} + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + differential_weight \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
        where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    r1, r2, r3, r4 = ic, ic, ic, ic
    while r1 != ic: r1 = rng.integers(len(pop))
    while r2 != ic and r2 != r1: r2 = rng.integers(len(pop))
    while r3 != ic and r3 != r2 and r3 != r1: r3 = rng.integers(len(pop))
    while r4 != ic and r4 != r1 and r4 != r2 and r4 != r3: r4 = rng.integers(len(pop))
    x = [pop[ic][i] + f * (pop[r1][i] - pop[r2][i]) + f * (
                pop[r3][i] - pop[r4][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_curr2best1(pop, ic, f, cr, rng, x_b=None, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation:
        Name: de/curr-to-best/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{i, G} + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + differential_weight \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
        where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        x_b (Individual): Current global best individual.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    r1, r2, r3, r4 = ic, ic, ic, ic
    while r1 != ic: r1 = rng.integers(len(pop))
    while r2 != ic and r2 != r1: r2 = rng.integers(len(pop))
    while r3 != ic and r3 != r2 and r3 != r1: r3 = rng.integers(len(pop))
    x = [
        pop[ic][i] + f * (x_b[i] - pop[r1][i]) + f * (pop[r2][i] - pop[r3][i]) if rng.random() < cr or i == j else
        pop[ic][i] for i in range(len(pop[ic]))]
    return np.asarray(x)


class DifferentialEvolution(OptimizationAlgorithm):
    r"""Implementation of Differential evolution algorithm.

    Algorithm:
         Differential evolution algorithm

    Date:
        2018

    Author:
        Uros Mlakar and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.

    Attributes:
        Name (List[str]): List of string of names for algorithm.
        differential_weight (float): Scale factor.
        crossover_probability (float): Crossover probability.
        strategy (Callable[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, Dict[str, Any]]): crossover and mutation strategy.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['DifferentialEvolution', 'DE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359."""

    def __init__(self, population_size=50, differential_weight=1, crossover_probability=0.8, strategy=cross_rand1, *args, **kwargs):
        """Initialize DifferentialEvolution.

        Args:
            population_size (Optional[int]): Population size.
            differential_weight (Optional[float]): Differential weight (differential_weight).
            crossover_probability (Optional[float]): Crossover rate.
            strategy (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, list], numpy.ndarray]]): Crossover and mutation strategy.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def set_parameters(self, population_size=50, differential_weight=1, crossover_probability=0.8, strategy=cross_rand1,
                       **kwargs):
        r"""Set the algorithm parameters.

        Args:
            population_size (Optional[int]): Population size.
            differential_weight (Optional[float]): Differential weight (differential_weight).
            crossover_probability (Optional[float]): Crossover rate.
            strategy (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, list], numpy.ndarray]]):
                Crossover and mutation strategy.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def get_parameters(self):
        r"""Get parameters values of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        See Also:
            * :func:`niapy.algorithms.Algorithm.get_parameters`

        """
        d = super().get_parameters()
        d.update({
            'differential_weight': self.differential_weight,
            'crossover_probability': self.crossover_probability,
            'strategy': self.strategy
        })
        return d

    def run_iteration(self, task, pop, fpop, xb, fxb, iters, **params):
        r"""Core function of Differential Evolution algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Current best individual.
            best_fitness (float): Current best individual function/fitness value.
            iters (int): Iteration number.
            **params (dict): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.evolve`
            * :func:`niapy.algorithms.basic.DifferentialEvolution.selection`
            * :func:`niapy.algorithms.basic.DifferentialEvolution.post_selection`

        """
        for i in range(len(pop)):
            xn = task.repair(self.strategy(pop, i, self.differential_weight, self.crossover_probability, self.rng, x_b=xb), rng=self.rng)
            fn = task.eval(xn)
            if fn < fpop[i]: 
                pop[i] = xn
                fpop[i] = fn
                if fn < fxb:
                    xb = np.copy(xn)
                    fxb = fn
        return pop, fpop, xb, fxb, {}


class DynNpDifferentialEvolution(DifferentialEvolution):
    r"""Implementation of Dynamic population size Differential evolution algorithm.

    Algorithm:
        Dynamic population size Differential evolution algorithm

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        p_max (int): Number of population reductions.
        rp (int): Small non-negative number which is added to value of generations.

    See Also:
        * :class:`niapy.algorithms.basic.DifferentialEvolution`

    """

    Name = ['DynNpDifferentialEvolution', 'dynNpDE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""No info"""

    def __init__(self, population_size=10, p_max=50, rp=3, *args, **kwargs):
        """Initialize DynNpDifferentialEvolution.

        Args:
            p_max (Optional[int]): Number of population reductions.
            rp (Optional[int]): Small non-negative number which is added to value of generations.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.p_max = p_max
        self.rp = rp

    def set_parameters(self, p_max=50, rp=3, **kwargs):
        r"""Set the algorithm parameters.

        Args:
            p_max (Optional[int]): Number of population reductions.
            rp (Optional[int]): Small non-negative number which is added to value of generations.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.p_max = p_max
        self.rp = rp

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        params = super().get_parameters()
        params.update({
            'p_max': self.p_max,
            'rp': self.rp
        })
        return params

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, iters, **params):
        r"""Core function of Differential Evolution algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Current best individual.
            best_fitness (float): Current best individual function/fitness value.
            iters (int): Number of algorithm iteration.
            **params (dict): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.evolve`
            * :func:`niapy.algorithms.basic.DifferentialEvolution.selection`
            * :func:`niapy.algorithms.basic.DifferentialEvolution.post_selection`

        """
        for i in range(len(population)):
            xn = self.strategy(population, i, self.differential_weight, self.crossover_probability, self.rng, x_b=best_x)
            fn = task.eval(xn)
            if fn < population_fitness[i]: 
                population[i] = xn
                population_fitness[i] = fn
                if best_x < best_fitness:
                    best_x = np.copy(xn)
                    best_fitness = fn
        gr = task.max_evals // (self.p_max * len(pop)) + self.rp
        new_np = len(population) // 2
        if (iters + 1) == gr and len(population) > 3:
            npop, nfs = [], []
            for i in range(new_np):
                if population[i] < population_fitness[i + new_np]:
                    npop.append(population[i])
                    nfs.append(population_fitness[i])
                else:
                    npop.append(population[i + new_np])
                    nfs.append(population_fitness[i + new_np])
        population = np.asarray(npop)
        population_fitness = np.asarray(nfs)
        return population, population_fitness, best_x, best_fitness, {}

