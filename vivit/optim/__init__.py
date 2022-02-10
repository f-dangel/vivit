"""Optimization methods using low-rank representations of the GGN/Fisher."""

from vivit.optim.computations import BaseComputations
from vivit.optim.damped_newton import DampedNewton
from vivit.optim.damping import ConstantDamping
from vivit.optim.gram_computations import GramComputations

__all__ = [
    "DampedNewton",
    "ConstantDamping",
    "BaseComputations",
    "GramComputations",
]
