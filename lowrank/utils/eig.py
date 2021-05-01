"""Eigenvalue decomposition utility functions."""

import torch


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

    Raises:
        ValueError: If ``input`` does not have dimension 2 or if it is not square.
        RuntimeError: If solver did not converge or if input or output tensor contains
            ``nan``s
    """
    if input.dim() != 2:
        raise ValueError("Input must be of dimension 2")
    if input.shape[0] != input.shape[1]:
        raise ValueError("Input must be square")
    if _has_nans(input):
        raise RuntimeError("Input has nans")

    def shift_diag(input, shift):
        """Shifts the diagonal of the square ``input`` tensor by ``shift``

        Args:
            input (torch.Tensor): 2d square tensor.
            shift (float): The shift applied to the diagonal of ``input``

        Returns:
            torch.Tensor: input with shifted diagonal
        """
        return input + torch.diag(shift * torch.ones(input.shape[0])).to(input.device)

    try:
        evals, evecs = input.symeig(eigenvectors=eigenvectors, upper=upper)
        return evals, evecs
    except RuntimeError:
        SHIFT = 1.0
        shifted_input = shift_diag(input, SHIFT)
        try:
            evals, evecs = shifted_input.symeig(eigenvectors=eigenvectors, upper=upper)
        except RuntimeError as e:
            e_msg = getattr(e, "message", repr(e))
            raise RuntimeError(f"{e_msg}")
        return evals - SHIFT, evecs


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
        e_msg = getattr(e, "message", repr(e))
        raise RuntimeError(f"{e_msg} Tensor contains NaNs: {_has_nans(input)}")

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
