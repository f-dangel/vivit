"""Utility functions for subsampling."""


def is_subset(subsampling, reference):
    """Return whether indices specified by ``subsampling`` are subset of the reference.

    Args:
        subsampling ([int] or None): Sample indices
        reference ([int] or None): Reference set.

    Returns:
        bool: Whether all indices are contained in the reference set.
    """
    if reference is None:
        return True
    elif subsampling is None and reference is not None:
        return False
    else:
        return set(subsampling).issubset(set(reference))


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
        [int] or None: Index mapping for samples to output index. ``None`` if the
            mapping is the identity.

    Raises:
         ValueError: If one of the requested samples is not contained in all samples.
    """
    if not is_subset(idx_samples, idx_all):
        raise ValueError(f"Requested samples {idx_samples} must be subset of {idx_all}")

    if idx_all is None:
        mapping = idx_samples
    else:
        mapping = [idx_all.index(sample) for sample in idx_samples]

    return mapping


def merge_subsamplings(subsampling, other):
    """Merge indices of sub-samplings, removing duplicates and sorting indices.

    Args:
        subsampling ([int] or None): Sub-sampling indices for use in a BackPACK
            extension as ``subsampling`` argument.
        other ([int] or None): Sub-sampling indices for use in a BackPACK
            extension as ``subsampling`` argument.

    Returns:
        [int]: Indices corresponding to the merged sub-samplings.
    """
    if subsampling is None or other is None:
        merged = None
    else:
        merged = sorted(set(subsampling).union(set(other)))

    return merged


def merge_multiple_subsamplings(*subsamplings):
    """Merge a sequence of sub-samplings, removing duplicates and sorting indices.

    Args:
        subsamplings ([[int] or None]): Sub-sampling sequence.

    Returns:
        [int]: Indices corresponding to the merged sub-samplings.

    Raises:
        ValueError: If no arguments are handed in
    """
    if len(subsamplings) == 0:
        raise ValueError("Expecting one or more inputs. Got {subsamplings}.")

    subsampling = []

    for other in subsamplings:
        subsampling = merge_subsamplings(subsampling, other)

    return subsampling


def merge_extensions(extension_subsampling_list):
    """Combine subsamplings of same extensions.

    Args:
        extension_subsampling_list ([tuple]): List of extension-subsampling
            pairs to be merged.

    Returns:
        dict: Keys are extension classes, values are subsamplings.
    """
    unique = {extension for (extension, _) in extension_subsampling_list}

    merged_subsamplings = {}

    for extension in unique:
        subsamplings = [
            sub for (ext, sub) in extension_subsampling_list if ext == extension
        ]
        merged_subsamplings[extension] = merge_multiple_subsamplings(*subsamplings)

    return merged_subsamplings
