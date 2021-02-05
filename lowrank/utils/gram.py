"""Utility functions to compute gram matrices."""

import torch


def pairwise_dot(tensor):
    """Compute pairwise scalar product. Pairs are the two leading dims."""
    out_dim = 2 * (tensor.shape[0] * tensor.shape[1],)

    # TODO Avoid flattening with more sophisticated einsum equation
    tensor_flat = tensor.flatten(start_dim=2)
    equation = "ijf,klf->ijkl"

    return torch.einsum(equation, tensor_flat, tensor_flat).reshape(out_dim)
