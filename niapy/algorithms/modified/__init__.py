# encoding=utf8
"""Implementation of modified nature-inspired algorithms."""

from niapy.algorithms.modified.hba import HybridBatAlgorithm
from niapy.algorithms.modified.hsaba import HybridSelfAdaptiveBatAlgorithm
from niapy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolution
from niapy.algorithms.modified.plba import ParameterFreeBatAlgorithm
from niapy.algorithms.modified.saba import (
    AdaptiveBatAlgorithm,
    SelfAdaptiveBatAlgorithm
)
from niapy.algorithms.modified.shade import (
    SuccessHistoryAdaptiveDifferentialEvolution,
    LpsrSuccessHistoryAdaptiveDifferentialEvolution
)

__all__ = [
    'HybridBatAlgorithm',
    'AdaptiveBatAlgorithm',
    'SelfAdaptiveBatAlgorithm',
    'ParameterFreeBatAlgorithm',
    'HybridSelfAdaptiveBatAlgorithm',
    'SelfAdaptiveDifferentialEvolution',
    'SuccessHistoryAdaptiveDifferentialEvolution',
    'LpsrSuccessHistoryAdaptiveDifferentialEvolution'
]
