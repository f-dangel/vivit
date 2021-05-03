"""Optimization methods using low-rank representations of the GGN/Fisher."""

from lowrank.optim.computations import BaseComputations
from lowrank.optim.damped_newton import DampedNewton
from lowrank.optim.damping import ConstantDamping

__all__ = [
    "DampedNewton",
    "ConstantDamping",
    "BaseComputations",
]
