"""Vectorized multiplication with the GGN(MC) symmetric factors (square roots)."""

import einops
import torch

from vivit.utils.gram import get_letters, pairwise2_dot, pairwise_dot


def V_t_mat_prod(mat_list, parameters, savefield, subsampling=None, flatten=False):
    """Multiply with the GGN(MC) matrix square root ``Vᵀ`` defined by ``parameters``.

    Shapes:
        The flattened ``Vᵀ`` has shape ``[C * N, D]``, and the GGN(MC) is ``V @ Vᵀ``.

        - ``N``: Batch size
        - ``C``: Model output dimension (number of classes) or number of MC samples
        - ``D``: Number of parameters

    Args:
        mat_list ([torch.Tensor]): List of tensors containing the matrix onto which
            ``Vᵀ`` is right-multiplied, separated layer-wise in same order as
            ``parameters``. For every parameter ``pᵢ`` with shape ``*ᵢ``, the entry
            ``matᵢ`` has shape ``[F, *ᵢ], i.e. same trailing dimension and a free
            leading dimension ``F``. Every slice along the free dimension is treated
            independently.
        parameters (iterable): Sequence of parameters whose GGN(MC) is used for
            multiplication with ``Vᵀ``.
        savefield (str): Attribute under which ``Vᵀ`` is stored in a parameter.
        flatten (bool): Whether to flatten the output dimensions ``[C, N]`` into
            ``[C * N]``. Default: ``False``.
        subsampling ([int]): Sample indices to be used of ``Vᵀ``. ``None`` uses all
            available samples.

    Returns:
        torch.Tensor: Result of multiplication. Has shape ``[F, C * N]`` if ``flatten``
            is ``True``, else ``[F, C, N]``.
    """
    # same free dimension
    _same_shape(mat_list, end=0)
    assert len(mat_list) == len(parameters), "Matrices must have length of parameters."

    result = sum(
        _param_V_t_mat_prod(p, mat, savefield, subsampling=subsampling)
        for p, mat in zip(parameters, mat_list)
    )

    if flatten:
        result = result.flatten(start_dim=1)

    return result


def _param_V_t_mat_prod(param, mat, savefield, subsampling=None):
    """Multiply with the GGN(MC) matrix square root ``Vᵀ`` defined by ``param``.

    Args:
        param (torch.nn.Parameter): Parameter defining ``Vᵀ``.
        mat (torch.Tensor): Matrix to be multiplied with ``Vᵀ``.
        savefield (str): Attribute under which ``Vᵀ`` is stored in ``param``.
        subsampling ([int]): Sample indices to be used of ``Vᵀ``. ``None`` uses all
            available samples.

    Returns:
        torch.Tensor: Result of the matrix-multiply ``Vᵀ @ mat``.
    """
    V_t = _get_V_t(param, savefield, subsampling=subsampling)
    start_dim = 2

    # build einsum equation
    letters = get_letters(V_t.dim() + 1)

    free_idx = letters[0]
    out_idx = letters[1 : start_dim + 1]
    sum_idx = letters[start_dim + 1 :]

    equation = f"{free_idx}{sum_idx},{out_idx}{sum_idx}->{free_idx}{out_idx}"

    return torch.einsum(equation, mat, V_t)


def V_mat_prod(mat, parameters, savefield, subsampling=None, concat=False):
    """Multiply with the GGN(MC) matrix square root ``V`` defined by ``parameters``.

    Shapes:
        Flattened ``V`` has shape ``[D, C * N]``, and the GGN(MC) is ``V @ Vᵀ``.

        - ``N``: Batch size
        - ``C``: Model output dimension (number of classes) or number of MC samples
        - ``D``: Number of parameters

    Args:
        mat ([torch.Tensor]): Matrix to be right multiplied by ``V``. Must have shape
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
        [torch.Tensor] or torch.Tensor: If ``concat`` is ``True``, the multiplication
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
        result = torch.cat(result, dim=start_dim)

    return result


def _get_V_t(param, savefield, subsampling=None):
    """Fetch the GGN(MC) matrix square root ``Vᵀ`` with active samples.

    Args:
        param (torch.nn.Parameter): Parameter defining ``Vᵀ``.
        savefield (str): Attribute under which ``Vᵀ`` is stored in a parameter.
        subsampling ([int]): Sample indices to be used of ``Vᵀ``. ``None`` uses all
            available samples.

    Returns:
        torch.Tensor: Sub-sampled ``Vᵀ`` tensor.
    """
    V_t = getattr(param, savefield)

    if subsampling is not None:
        V_t = V_t[:, subsampling]

    return V_t


def V_param_mat_prod(param, mat, savefield, subsampling=None):
    """Multiply with the GGN(MC) matrix square root ``V`` defined by ``param``.

    Args:
        param (torch.nn.Parameter): Parameter defining ``Vᵀ``.
        mat (torch.Tensor): Matrix to be multiplied with ``V``.
        savefield (str): Attribute under which ``Vᵀ`` is stored in ``param``.
        subsampling ([int]): Sample indices to be used of ``Vᵀ``. ``None`` uses all
            available samples.

    Returns:
        torch.Tensor: Result of the matrix-multiply ``V @ mat``.
    """
    V_t = _get_V_t(param, savefield, subsampling=subsampling)
    start_dim = 2

    # build einsum equation
    letters = get_letters(V_t.dim() + 1)

    free_idx = letters[0]
    sum_idx = letters[1 : start_dim + 1]
    out_idx = letters[start_dim + 1 :]

    equation = f"{free_idx}{sum_idx},{sum_idx}{out_idx}->{free_idx}{out_idx}"

    return torch.einsum(equation, mat, V_t)


def _same_shape(tensor_list, start=0, end=-1):
    """Return whether tensors have same (sub-)shapes.

    Args:
        tensor_list ([torch.Tensor]): List of tensors whose shapes are compared.
        start (int, optional): The dimension to start comparing shapes.
        end (int, optional): The dimension until (including) shapes are compared.

    Raises:
        ValueError: If the tensors don't have identical (sub-)shapes.
    """

    def _slice(shape):
        """Get relevant parts of the shape for comparison.

        Args:
            shape (torch.Size): Shape of a ``torch.Tensor``.

        Returns:
            torch.Size: Relevant sub-shape.
        """
        return shape[start:] if end == -1 else shape[start : end + 1]

    unique = {_slice(tensor.shape) for tensor in tensor_list}

    if len(unique) != 1:
        raise ValueError(
            f"Got non-unique shapes comparing dims {start} to including {end}: {unique}"
        )


def V_t_V(parameters, savefield, subsampling=None, flatten=False):
    """Compute the Gram matrix ``Vᵀ V``.

    Args:
        parameters (iterable): Sequence of parameters whose GGN(MC) factors are used.
        savefield (str): Attribute under which ``Vᵀ`` and ``V`` are stored.
        subsampling ([int]): Sample indices to be used of ``Vᵀ, V``. ``None`` uses all
            available samples.
        flatten (bool): Whether to flatten the output dimensions ``[C, N]`` into
            ``[C * N]``. Default: ``False``.

    Returns:
        torch.Tensor: Gram matrix. If ``flatten`` is ``True``, the shape is
            ``[C, N, C, N]``, else ``[C * N, C * N]``.
    """
    result = sum(
        _param_V_t_V(p, savefield, subsampling=subsampling) for p in parameters
    )

    if flatten:
        result = einops.rearrange("c n d m -> (c n) (d m)", result)

    return result


def _param_V_t_V(param, savefield, subsampling=None):
    """Compute the Gram matrix ``Vᵀ V``.

    Args:
        param (torch.nn.Parameter): Parameter defining ``Vᵀ`` and ``V``.
        savefield (str): Attribute under which ``Vᵀ`` and ``V`` are stored.
        subsampling ([int]): Sample indices to be used of ``Vᵀ, V``. ``None`` uses all
            available samples.

    Returns:
        torch.Tensor: Gram matrix of shape ``[C, N, C, N]``.
    """
    start_dim = 2
    assert savefield in ["sqrt_ggn_mc", "sqrt_ggn_exact"], "Only GGN factors"

    V_t = _get_V_t(param, savefield, subsampling=subsampling)

    return pairwise_dot(V_t, start_dim=start_dim, flatten=False)


def V1_t_V2(parameters, savefield1, savefield2, subsampling1=None, subsampling2=None):
    """Compute the overlap ``V₁ᵀ V₂`` between two GGN factors.

    Args:
        parameters (iterable): Sequence of parameters whose GGN(MC) factors are used.
        savefield1 (str): Attribute under which ``V₁ᵀ`` is stored.
        savefield2 (str): Attribute under which ``V₂ᵀ`` is stored.
        subsampling1 ([int], optional): Sample indices to be used of ``V₁ᵀ``. ``None``
            uses all available samples.
        subsampling2 ([int], optional): Sample indices to be used of ``V₂ᵀ``. ``None``
            uses all available samples.

    Returns:
        torch.Tensor: Overlap matrix of shape ``[C₁, N₁, C₂, N₂]``, where ``Cₙ`` and
            ``Nₙ`` are the classes and samples used to represent ``Vₙᵀ``.
    """
    return sum(
        _param_V1_t_V2(
            p,
            savefield1,
            savefield2,
            subsampling1=subsampling1,
            subsampling2=subsampling2,
        )
        for p in parameters
    )


def _param_V1_t_V2(param, savefield1, savefield2, subsampling1=None, subsampling2=None):
    """Compute overlap ``V₁ᵀ V₂`` between two GGN factors restricted to one parameter.

    Args:
        param (torch.nn.Parameter): Parameter whose GGN(MC) factors are used.
        savefield1 (str): Attribute under which ``V₁ᵀ`` is stored.
        savefield2 (str): Attribute under which ``V₂ᵀ`` is stored.
        subsampling1 ([int], optional): Sample indices to be used of ``V₁ᵀ``. ``None``
            uses all available samples.
        subsampling2 ([int], optional): Sample indices to be used of ``V₂ᵀ``. ``None``
            uses all available samples.

    Returns:
        torch.Tensor: Overlap matrix of shape ``[C₁, N₁, C₂, N₂]``, where ``Cₙ`` and
            ``Nₙ`` are the classes and samples used to represent ``Vₙᵀ``.
    """
    start_dim = 2
    for savefield in [savefield1, savefield2]:
        assert savefield in ["sqrt_ggn_mc", "sqrt_ggn_exact"], "Only GGN factors"

    V1_t = _get_V_t(param, savefield1, subsampling=subsampling1)
    V2_t = _get_V_t(param, savefield2, subsampling=subsampling2)

    return pairwise2_dot(V1_t, V2_t, start_dim=start_dim)
