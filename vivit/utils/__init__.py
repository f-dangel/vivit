"""Utility functions."""

from typing import Optional

from torch import Tensor


def delete_savefield(param: Tensor, savefield: str, verbose: Optional[bool] = False):
    """Delete attribute of a parameter.

    Args:
        param: Parameter.
        savefield: Name of removed attribute.
        verbose: Print action to command line. Default: ``False``.
    """
    if verbose:
        print(f"Param {id(param)}: Delete '{savefield}'")

    delattr(param, savefield)
