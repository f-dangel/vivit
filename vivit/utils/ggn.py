"""Vectorized multiplication with the GGN(MC) symmetric factors (square roots)."""

from typing import List

from torch import Tensor, cat, einsum
from torch.nn import Parameter

from vivit.utils.gram import get_letters


def V_mat_prod(mat, parameters, savefield, subsampling=None, concat=False):
    """Multiply with the GGN(MC) matrix square root ``V`` defined by ``parameters``.

    Shapes:
        Flattened ``V`` has shape ``[D, C * N]``, and the GGN(MC) is ``V @ Vᵀ``.

        - ``N``: Batch size
        - ``C``: Model output dimension (number of classes) or number of MC samples
        - ``D``: Number of parameters

    Args:
        mat ([Tensor]): Matrix to be right multiplied by ``V``. Must have shape
            ``[F, C, N]``. ``F`` is a free leading dimension. Every slice along it is
            multiplied independently.
        parameters (iterable): Sequence of parameters whose GGN(MC) is used for
            multiplication with ``V``.
        savefield (str): Attribute under which ``Vᵀ`` is stored in a parameter.
        concat (bool, optional): Whether to flatten and concatenate the result over
            parameters. Default: ``False``.
        subsampling ([int]): Sample indices to be used of ``Vᵀ``. ``None`` uses all
            available samples.

    Returns:
        [Tensor] or Tensor: If ``concat`` is ``True``, the multiplication
            results are flattened and concatenated into a ``[F, D]`` tensor. Otherwise,
            a list of tensors with same length as ``parameters`` and shape ``[F, *ᵢ]``
            is returned (``*ᵢ`` is the shape of parameter ``i``).
    """
    assert mat.dim() == 3, f"mat must be [F, C, N]. Got {mat.dim()} dimensions."

    result = [
        V_param_mat_prod(p, mat, savefield, subsampling=subsampling) for p in parameters
    ]

    if concat:
        start_dim = 1
        result = [res.flatten(start_dim=start_dim) for res in result]
        result = cat(result, dim=start_dim)

    return result


def _get_V_t(param, savefield, subsampling=None):
    """Fetch the GGN(MC) matrix square root ``Vᵀ`` with active samples.

    Args:
        param (Parameter): Parameter defining ``Vᵀ``.
        savefield (str): Attribute under which ``Vᵀ`` is stored in a parameter.
        subsampling ([int]): Sample indices to be used of ``Vᵀ``. ``None`` uses all
            available samples.

    Returns:
        Tensor: Sub-sampled ``Vᵀ`` tensor.
    """
    V_t = getattr(param, savefield)

    if subsampling is not None:
        V_t = V_t[:, subsampling]

    return V_t


def V_param_mat_prod(
    param: Parameter, mat: Tensor, savefield: str, subsampling: List[int] = None
):
    """Multiply with the GGN(MC) matrix square root ``V`` defined by ``param``.

    Args:
        param: Parameter defining ``Vᵀ``.
        mat: Matrix to be multiplied with ``V``.
        savefield: Attribute under which ``Vᵀ`` is stored in ``param``.
        subsampling: Sample indices to be used of ``Vᵀ``. ``None`` uses all
            available samples.

    Returns:
        Result of the matrix-multiply ``V @ mat``.
    """
    V_t = _get_V_t(param, savefield, subsampling=subsampling)
    start_dim = 2

    return Vmp(V_t, mat, start_dim)


def Vmp(V_t: Tensor, mat: Tensor, start_dim: int) -> Tensor:
    """Vectorized application of ``V`` to a vector.

    Args:
        V_t: Tensor representation of ``Vᵀ``. Shape ``[*start_dims, *param.shape()]``.
        mat: Stacked vectors onto which ``V`` will be applied. Shape
            ``[V.shape[0], *start_dims]
        start_dim: Index that indicates the first axis for a vector in ``mat`` to be
            contracted.

    Returns:
        Result of the matrix-multiply ``V @ mat``.
    """
    letters = get_letters(V_t.dim() + 1)

    free_idx = letters[0]
    sum_idx = letters[1 : start_dim + 1]
    out_idx = letters[start_dim + 1 :]

    equation = f"{free_idx}{sum_idx},{sum_idx}{out_idx}->{free_idx}{out_idx}"

    return einsum(equation, mat, V_t)
