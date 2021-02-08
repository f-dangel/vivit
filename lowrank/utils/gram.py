"""Utility functions to compute gram matrices."""

import torch


def pairwise_dot(tensor, start_dim=1):
    """Compute pairwise scalar product. Pairs are determined by ``start_dim``."""
    out_dim = int(torch.prod(torch.Tensor(tuple(tensor.shape[:start_dim]))))
    out_shape = (out_dim, out_dim)

    # build einsum equation
    letters = get_letters(start_dim + tensor.dim())
    out1_idx = letters[:start_dim]
    out2_idx = letters[start_dim : 2 * start_dim]
    sum_idx = letters[2 * start_dim :]

    equation = f"{out1_idx}{sum_idx},{out2_idx}{sum_idx}->{out1_idx}{out2_idx}"

    return torch.einsum(equation, tensor, tensor).reshape(out_shape)


def get_letters(num_letters):
    """Return a list of ``num_letters`` unique letters."""
    MAX_LETTERS = 26

    if num_letters > MAX_LETTERS:
        raise ValueError(f"Requested too many letters {num_letters}>{MAX_LETTERS}")

    return "".join(chr(ord("a") + num) for num in range(num_letters))
