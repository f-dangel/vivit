"""Utility functions to compute gram matrices."""

import math

import torch


def pairwise_dot(tensor, start_dim=1, flatten=True):
    """Compute pairwise scalar product. Pairs are determined by ``start_dim``.

    Args:
        tensor (torch.Tensor): A tensor whose slices, depending on ``start_dim``,
            are vectors whose pairwise scalar product will be computed.
        start_dim (int): Leading dimensions that define the set of vectors that
            whose pairwise scalar product will be computed.
        flatten (bool): Return the result as square-shaped matrix, i.e. flatten
            the index set of vectors that were dotted. If ``False``, return the
            unflattened tensor of dimension ``2 * start_dim``.

    Returns:
        torch.Tensor: If ``reshape=True`` a square matrix of shape ``[∏ᵢ dᵢ, ∏ᵢ dᵢ]``
            where ``i`` ranges from ``0`` to ``start_dim - 1`` and ``dᵢ`` is the
            ``i``th dimension of ``tensor``.

            If ``reshape=False`` a tensor of shape ``[*(dᵢ), *(dᵢ)]``
            where ``i`` ranges from ``0`` to ``start_dim - 1`` and ``dᵢ`` is the
            ``i``th dimension of ``tensor``.
    """
    # build einsum equation
    letters = get_letters(start_dim + tensor.dim())
    out1_idx = letters[:start_dim]
    out2_idx = letters[start_dim : 2 * start_dim]
    sum_idx = letters[2 * start_dim :]

    equation = f"{out1_idx}{sum_idx},{out2_idx}{sum_idx}->{out1_idx}{out2_idx}"
    result = torch.einsum(equation, tensor, tensor)

    if flatten:
        result = reshape_as_square(result)

    return result


def get_letters(num_letters):
    """Return a list of ``num_letters`` unique letters."""
    MAX_LETTERS = 26

    if num_letters > MAX_LETTERS:
        raise ValueError(f"Requested too many letters {num_letters}>{MAX_LETTERS}")

    return "".join(chr(ord("a") + num) for num in range(num_letters))


def reshape_as_square(tensor):
    """Rearrange the elements of an arbitrary tensor into a square matrix.

    Args:
        tensor (torch.Tensor): Any tensor.

    Returns:
        torch.Tensor: A square-matrix containing the same elements as ``tensor``.
    """
    dim = int(math.sqrt(tensor.numel()))

    return tensor.reshape(dim, dim)
