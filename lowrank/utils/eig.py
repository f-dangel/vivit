"""Eigenvalue decomposition utility functions."""

import torch


def symeig(input, eigenvectors=False, upper=True, atol=1e-7, rtol=1e-5):
    """Compute EVals and EVecs of a matrix. Discard pairs with EVal ≈ 0.

    This is a wrapper around ``torch.symeig`` plus filtering of EVal/EVec pairs
    numerically close to zero. Use ``torch.symeig`` if you want all EVal/Evecs.

    Args:
        atol (float): Absolute tolerance to detect zero EVals.
        rtol (float): Relative tolerance to detect zero EVals.

    Returns:
        (torch.Tensor, torch.Tensor): First tensor of one dimension contains
            eigenvalues. Second tensor holds associated eigenvectors stored columnwise,
            i.e. ``evecs[:, i]`` is eigenvector with eigenvalue ``evals[i]``.
    """
    if input.dim() != 2:
        raise ValueError("Input must be of dimension 2")

    evals, evecs = input.symeig(eigenvectors=eigenvectors, upper=upper)

    return remove_zero_evals(evals, evecs, atol=atol, rtol=rtol)


def remove_zero_evals(evals, evecs, atol=1e-7, rtol=1e-5):
    """Remove (EVal, EVec) pairs if EVal ≈ 0.

    ``evals`` and ``evecs`` are assumed to be output of ``torch.symeig``.

    Args:
        evals (torch.Tensor): 1d tensor of eigenvalues.
        evecs (torch.Tensor): 2d or empty tensor of eigenvalues.
        atol (float): Absolute tolerance to detect zero EVals.
        rtol (float): Relative tolerance to detect zero EVals.

    Returns
        (torch.Tensor, torch.Tensor): Filtered EVals and EVecs.
    """
    nonzero = torch.isclose(
        evals, torch.zeros_like(evals), rtol=rtol, atol=atol
    ).logical_not()

    evals = evals[nonzero]

    if evecs.numel() != 0:
        evecs = evecs[:, nonzero]

    return evals, evecs
