"""Utility functions to deal with parameter groups."""

from typing import Dict, List, Union


def check_key_exists(param_groups: List[Dict], key: str):
    """Check if all groups specify the key.

    Args:
        param_groups: Parameter groups that define the GGN block structure.
        key: The key to check for in each group.

    Raises:
        ValueError: If any group does not specify the key.
    """
    if any(key not in group.keys() for group in param_groups):
        raise ValueError(f"At least one group is not specifying '{key}'.")


def check_unique_params(param_groups: List[Dict]):
    """Check that each parameter is assigned to one group only.

    Args:
        param_groups: Parameter groups that define the GGN block structure.

    Raises:
        ValueError: If a parameter occurs in multiple groups.
    """
    params_ids = []
    for group in param_groups:
        params_ids += [id(p) for p in group["params"]]

    if len(set(params_ids)) != len(params_ids):
        raise ValueError("At least one parameter is in more than one group.")


def check_subsampling_unique(subsampling: Union[None, List[int]]):
    """Check that sub-sampling contains unique sample indices.

    Args:
        subsampling: Indices of active samples used for a computation. ``None``
            uses the full mini-batch.

    Raises:
        ValueError: If the same index occurs more than once.
    """
    if subsampling is not None:
        if len(set(subsampling)) != len(subsampling):
            raise ValueError("Detected repeated index in subsampling.")
