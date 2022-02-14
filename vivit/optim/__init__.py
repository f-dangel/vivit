"""Optimization methods using low-rank representations of the GGN/Fisher."""

from vivit.optim.computations import BaseComputations
from vivit.optim.damped_newton import DampedNewton
from vivit.optim.damping import ConstantDamping
from vivit.optim.directional_derivatives import DirectionalDerivativesComputation

__all__ = [
    "DampedNewton",
    "ConstantDamping",
    "BaseComputations",
    "DirectionalDerivativesComputation",
]
