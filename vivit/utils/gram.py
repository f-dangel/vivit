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
        torch.Tensor: If ``flatten=True`` a square matrix of shape ``[∏ᵢ dᵢ, ∏ᵢ dᵢ]``
            where ``i`` ranges from ``0`` to ``start_dim - 1`` and ``dᵢ`` is the
            ``i``th dimension of ``tensor``.

            If ``flatten=False`` a tensor of shape ``[*(dᵢ), *(dᵢ)]``
            where ``i`` ranges from ``0`` to ``start_dim - 1`` and ``dᵢ`` is the
            ``i``th dimension of ``tensor``.
    """
    result = partial_contract(tensor, tensor, (start_dim, start_dim))

    if flatten:
        result = reshape_as_square(result)

    return result


def pairwise2_dot(tensor, other, start_dim=1):
    """Compute pairwise scalar products between two tensors.

    Args:
        tensor (torch.Tensor): A tensor whose vector slices are paired with
            slices of ``other``.
        other (torch.Tensor): A tensor whose vector slices are paired with
            slices of ``tensor``.
        start_dim (int): Start dimension of contraction for scalar product.

    Returns:
        torch.Tensor: A tensor of shape ``[*(tensorᵢ), *(otherⱼ)]`` where `i, j``
            range from ``0`` to ``start_dim - 1``. ``tensorᵢ`` is the ``i``th
            dimension of ``tensor``, and ``otherⱼ`` is the ``j`` dimension of
            ``other``.
    """
    return partial_contract(tensor, other, (start_dim, start_dim))


def get_letters(num_letters):
    """Return a list of ``num_letters`` unique letters.

    Args:
        num_letters (int): Number of requested unique letters.

    Returns:
        str: Unique letters, concatenated as a string.

    Raises:
        ValueError: If the number of requested letters exceeds the alphabet size.
    """
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


def compute_gram_mat(parameters, savefield, start_dim, flatten=True):
    """Compute the Gram matrix from a BackPACK quantity.

    Fetches the quantity in ``savefield`` and computes pairwise scalar products.
    Different vectors are given by the flattened dimensions, up to ``start_dim``
    of the quantity. Results are summed over parameters.

    Args:
        parameters (iterable): Iterable to traverse the parameters for which BackPACK
            quantities were computed during a backward pass.
        savefield (str): The attribute name under which the BackPACK quantity is saved.
            If ``p`` is a parameter in ``parameters``, then ``p.savefield`` will be
            used to construct the Gram matrix.
        start_dim (int): Dimensions until ``start_dim`` are detected as vectors that
            form the Gram matrix, while the dimensions after ``start_dim`` will be
            contracted in the dot product. Should be ``1`` for individual gradient Gram
            matrices and 2 for Gram matrices related to the GGN.
        flatten (bool): Return the result as square-shaped matrix, i.e. flatten
            the index set of vectors that were dotted. If ``False``, return the
            unflattened tensor of dimension ``2 * start_dim``.

    Returns:
        torch.Tensor: If ``flatten=True`` a square matrix of shape ``[∏ᵢ dᵢ, ∏ᵢ dᵢ]``
            where ``i`` ranges from ``0`` to ``start_dim - 1`` and ``dᵢ`` is the
            ``i``th dimension of ``p.savefield``. For individual gradient Gram
            matrices, this shape is ``[N, N]``, and for GGN Gram matrices
            ``[N * C, N * C]``.

            If ``flatten=False``, the index set of Gram vectors will not be flattened.
            For individual gradient Gram matrices, the returned shape is ``[N, N]``,
            and for GGN Gram matrices ``[N, C, N, C]``.
    """
    gram = None

    for p in parameters:
        gram_p = pairwise_dot(
            getattr(p, savefield), start_dim=start_dim, flatten=flatten
        )

        if gram is None:
            gram = gram_p
        else:
            gram += gram_p

    return gram


def sqrt_gram_mat_prod(mat, parameters, savefield, start_dim, concat=False):
    """Multiply columns of ``mat`` with the Gram matrix square root ``U``.

    The Gram matrix is ``Uᵀ U``.

    Args:
        mat (torch.Tensor): A matrix-shaped tensor whose columns will be multiplied
            with the Gram matrix square root.
        parameters (iterable): Iterable object to traverse the parameters that con-
            tribute to the Gram matrix square root.
        savefield (str): Attribute field from which the current parameter's Gram matrix
            square root will be fetched.
        start_dim (int): ``1`` for gradient covariance, ``2`` for GGN matrices.
        concat (bool): If ``True``, flatten the parameter dimension and concatenate
            results over all parameters.

    Returns:
        list(torch.Tensor): If ``concat=False``. A list of tensors
            containing the matrix-multiply result. Since application of ``U`` gives
            results of parameter space dimension, the resulting vectors are split
            according to the ``parameters``. If ``mat`` has shape ``[I, J]``, then
            the result according to parameter ``p`` has shape ``[*p.shape, J]``,
            i.e. the leading dimensions are the parameter spaces' degrees of
            freedom, and the last dimension indicates the column space of ``mat``.

        torch.Tensor: If ``concat=True``, every element of the above list is flattened
            up to the column space of ``mat``, then concatenated along the flattened
            parameter dimension.

    Raises:
        NotImplementedError: If ``mat`` does not have dimension 2.
    """
    if mat.dim() != 2:
        raise NotImplementedError("Can only multiply with matrices")

    def mat_prod(p):
        """Multiply columns of ``mat`` with ``p``'s Gram matrix square root ``U``.

        Args:
            p (torch.nn.Parameter): Parameter whose ``U`` will be used.

        Returns:
            torch.Tensor: Result of the matrix-multiply ``U @ mat``.
        """
        gram_sqrt = getattr(p, savefield).flatten(end_dim=start_dim - 1)

        letters = get_letters(mat.dim() + gram_sqrt.dim() - 1)
        sum_idx = letters[0]
        out1_idx = letters[1:-1]
        out2_idx = letters[-1]

        equation = f"{sum_idx}{out1_idx},{sum_idx}{out2_idx}->{out1_idx}{out2_idx}"

        return torch.einsum(equation, gram_sqrt, mat)

    result = [mat_prod(p) for p in parameters]

    if concat:
        result = torch.cat([res.flatten(end_dim=res.dim() - 2) for res in result])

    return result


def partial_contract(tensor, other, start_dims):
    """Compute partial scalar products. Contract dimensions from ``start_dims``.

    Args:
        tensor (torch.Tensor): Left input to the scalar product.
        other (torch.Tensor): Right input to the scalar product.
        start_dims ([int, int]): List holding the dimensions at which the dot
            product contractions starts.

    Returns:
        torch.Tensor: Pair-wise scalar products. Has ``sum(start_dims)`` dimensions.

    Raises:
        ValueError: If the contracted dimensions don't match.
    """
    out_dim1, out_dim2 = start_dims
    feature_dim1, feature_dim2 = tensor.dim() - out_dim1, other.dim() - out_dim2

    if feature_dim1 != feature_dim2:
        raise ValueError("Trailing dimensions don't match.")

    letters = [out_dim1, out_dim2, feature_dim1]
    out_idx1, out_idx2, sum_idx = split_list(get_letters(sum(letters)), letters)

    equation = f"{out_idx1}{sum_idx},{out_idx2}{sum_idx}->{out_idx1}{out_idx2}"

    return torch.einsum(equation, tensor, other)


def split_list(sequence, lengths):
    """Consecutively split a list into sub-lists.

    Args:
        sequence (list): Sequence that will be split.
        lengths ([int]): List containing the sub-list lengths. The sub-list lengths
            must sum up to the length of ``sequence``.

    Yields:
        list: Next sublist

    Raises:
        ValueError: If the sub-list lengths don't sum to the total sequence length.
    """
    if len(sequence) != sum(lengths):
        raise ValueError("Sub-list lengths don't sum to length of the full list.")

    idx = 0

    for length in lengths:
        yield sequence[idx : idx + length]
        idx += length
