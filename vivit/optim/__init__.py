"""Optimization methods using low-rank representations of the GGN/Fisher."""

from vivit.optim.directional_damped_newton import DirectionalDampedNewtonComputation
from vivit.optim.directional_derivatives import DirectionalDerivativesComputation

__all__ = [
    "DirectionalDerivativesComputation",
    "DirectionalDampedNewtonComputation",
]
