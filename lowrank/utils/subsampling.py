"""Utility functions for subsampling."""


def sample_output_mapping(idx_samples, idx_all):
    """Return access indices for sub-sampled BackPACK quantities.

    Args:
        idx_samples ([int]): Mini-batch sample indices of samples to be accessed.
        idx_all ([int] or None): Sub-sampling indices used in the BackPACK extension
            whose savefield is being accessed. ``None`` signifies the entire batch
            was used.

    Example:
        Let's say we want to compute individual gradients for samples 0, 2, 3 from a
        mini-batch with ``N = 5`` samples. Those samples are described by the indices

        ``samples = [0, 1, 2, 3, 4]``

        Calling ``BatchGrad`` with ``subsampling = [0, 2, 3]``, will result in

        ``grad_batch = [∇f₀, ∇f₂, ∇f₃]``

        To access the gradient for sample 3, we need a mapping:

        ``mapping = [2]``

        Then, ``[∇f₃] = grad_batch[mapping]``.

    Returns:
        [int]: Index mapping for samples to output index.
    """
    assert idx_samples is not None

    if idx_all is None:
        mapping = idx_samples
    else:
        mapping = [idx_all.index(sample) for sample in idx_samples]

    return mapping


def merge_subsamplings(subsampling1, subsampling2):
    """Merge indices of sub-samplings, removing duplicates and sorting indices.

    Args:
        subsampling1 ([int] or None): Sub-sampling indices for use in a BackPACK
            extension as ``subsampling`` argument.
        subsampling2 ([int] or None): Sub-sampling indices for use in a BackPACK
            extension as ``subsampling`` argument.

    Returns:
        [int]: Indices corresponding to the merged sub-samplings.
    """
    merged = None

    if subsampling1 is not None and subsampling2 is not None:

        for subsampling in (subsampling1, subsampling2):
            assert isinstance(subsampling, list), "Must be list"

        merged = sorted(list(set(subsampling1 + subsampling2)))

    return merged
