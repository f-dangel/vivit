"""``vivit`` library API."""


from vivit import extensions
from vivit.linalg.eigh import EighComputation
from vivit.linalg.eigvalsh import EigvalshComputation
from vivit.optim.directional_damped_newton import DirectionalDampedNewtonComputation
from vivit.optim.directional_derivatives import DirectionalDerivativesComputation

__all__ = [
    "extensions",
    "optim",
    "EigvalshComputation",
    "EighComputation",
    "DirectionalDerivativesComputation",
    "DirectionalDampedNewtonComputation",
]
