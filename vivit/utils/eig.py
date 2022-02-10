"""Eigenvalue decomposition utility functions."""

import torch


# TODO Deprecate this method in favor of symeig_psd
def stable_symeig(input, eigenvectors=False, upper=True):
    """Compute EVals and EVecs of a matrix.

    This is a wrapper around ``torch.symeig``. If ``torch.symeig`` fails to compute
    the eigenvalues and -vectors, we shift the diagonal of the input by ``1`` to
    decrease the condition number. Then ``torch.symeig`` is called on this matrix and
    we obtain the eigenvalues by substracting ``1`` again.

    Args:
        input (torch.Tensor): 2d symmetric tensor.
        eigenvectors (bool): Whether eigenvectors should be computed.
        upper(bool): Whether to consider upper-triangular or lower-triangular
            region of the matrix.

    Returns:
        (torch.Tensor, torch.Tensor): First tensor of one dimension contains
            eigenvalues. Second tensor holds associated eigenvectors stored columnwise,
            i.e. ``evecs[:, i]`` is eigenvector with eigenvalue ``evals[i]``.
    """
    try:
        eig = input.symeig(eigenvectors=eigenvectors, upper=upper)
    except RuntimeError:
        eig = symeig_psd(
            input,
            eigenvectors=eigenvectors,
            upper=upper,
            shift=1.0,
            shift_inplace=False,
        )
    return eig


def symeig_psd(input, eigenvectors=False, upper=True, shift=0.0, shift_inplace=False):
    """Compute EVals and EVecs of a positive semi-definite symmetric matrix.

    This is a wrapper around ``torch.symeig``. It shifts the input's diagonal to
    improve its condition number. This avoids convergence problems with
    ``torch.symeig`` for ill-conditioned positive semi-definite matrices.

    Args:
        input (torch.Tensor): 2d symmetric tensor.
        eigenvectors (bool): Whether eigenvectors should be computed.
        upper(bool): Whether to consider upper-triangular or lower-triangular
            region of the matrix.
        shift (float, optional): The shift applied to the diagonal of ``input``.
            Default value: ``0.0``.
        shift_inplace (bool, optional): Shift the input inplace. If ``False``, copy the
            input before shifting. Default value: ``False``.

    Returns:
        (torch.Tensor, torch.Tensor): First tensor of one dimension contains
            eigenvalues. Second tensor holds associated eigenvectors stored columnwise,
            i.e. ``evecs[:, i]`` is eigenvector with eigenvalue ``evals[i]``.

    Raises:
        ValueError: If ``input`` does not have dimension 2.
        RuntimeError: If solver did not converge or input contains ``nan``s.
    """
    if input.dim() != 2:
        raise ValueError(f"Input must have dimension 2. Got {input.dim()}.")

    input = shift_diag(input, shift, inplace=shift_inplace)

    try:
        evals, evecs = input.symeig(eigenvectors=eigenvectors, upper=upper)
    except RuntimeError as e:
        raise RuntimeError(f"Tensor contains NaNs: {_has_nans(input)}") from e

    # shift back
    if shift_inplace:
        input = shift_diag(input, -shift, inplace=shift_inplace)

    evals -= shift

    return evals, evecs


def shift_diag(input, shift, inplace=False):
    """Shifts the diagonal of the square ``input`` tensor by ``shift``.

    Args:
        input (torch.Tensor): 2d tensor.
        shift (float): The shift applied to the diagonal of ``input``
        inplace (bool, optional): Modify the tensor inplace. If ``False``, copy the
            input before shifting. Default value: ``False``.

    Returns:
        torch.Tensor: Input with shifted diagonal.
    """
    if shift == 0.0:
        return input

    if inplace:
        result = input
    else:
        result = input.clone()

    min_dim = min(input.shape)
    result[range(min_dim), range(min_dim)] += shift

    return result


def symeig(input, eigenvectors=False, upper=True, atol=1e-7, rtol=1e-5):
    """Compute EVals and EVecs of a matrix. Discard pairs with EVal ≈ 0.

    This is a wrapper around ``torch.symeig`` plus filtering of EVal/EVec pairs
    numerically close to zero. Use ``torch.symeig`` if you want all EVal/Evecs.

    Args:
        input (torch.Tensor): 2d symmetric tensor.
        eigenvectors (bool): Whether eigenvectors should be computed.
        upper(bool): Whether to consider upper-triangular or lower-triangular
            region of the matrix.
        atol (float): Absolute tolerance to detect zero EVals.
        rtol (float): Relative tolerance to detect zero EVals.

    Returns:
        (torch.Tensor, torch.Tensor): First tensor of one dimension contains
            eigenvalues. Second tensor holds associated eigenvectors stored columnwise,
            i.e. ``evecs[:, i]`` is eigenvector with eigenvalue ``evals[i]``.

    Raises:
        ValueError: If ``input`` does not have dimension 2.
        RuntimeError: If solver did not converge.
    """
    if input.dim() != 2:
        raise ValueError("Input must be of dimension 2")

    try:
        evals, evecs = input.symeig(eigenvectors=eigenvectors, upper=upper)
    except RuntimeError as e:
        raise RuntimeError(f"Tensor contains NaNs: {_has_nans(input)}") from e

    return remove_zero_evals(evals, evecs, atol=atol, rtol=rtol)


def remove_zero_evals(evals, evecs, atol=1e-7, rtol=1e-5):
    """Remove (EVal, EVec) pairs if EVal ≈ 0.

    ``evals`` and ``evecs`` are assumed to be output of ``torch.symeig``.

    Args:
        evals (torch.Tensor): 1d tensor of eigenvalues.
        evecs (torch.Tensor): 2d or empty tensor of eigenvalues.
        atol (float): Absolute tolerance to detect zero EVals.
        rtol (float): Relative tolerance to detect zero EVals.

    Returns:
        (torch.Tensor, torch.Tensor): Filtered EVals and EVecs.
    """
    nonzero = torch.isclose(
        evals, torch.zeros_like(evals), rtol=rtol, atol=atol
    ).logical_not()

    evals = evals[nonzero]

    if evecs.numel() != 0:
        evecs = evecs[:, nonzero]

    return evals, evecs


def _has_nans(tensor):
    """Return whether a tensor contains NaNs.

    Args:
        tensor (torch.Tensor): Tensor to be checked.

    Returns:
        bool: ``True`` if ``tensor`` contains NaNs, else ``False``.
    """

    return torch.any(torch.isnan(tensor))
