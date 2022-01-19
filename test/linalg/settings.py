"""Test cases for linear algebra operations with ViViT."""

from test.optim.settings import (
    IDS_REDUCTION_MEAN,
    PARAM_BLOCKS_FN,
    PARAM_BLOCKS_FN_IDS,
    PROBLEMS_REDUCTION_MEAN,
)
from typing import List

from torch import Tensor

PROBLEMS = PROBLEMS_REDUCTION_MEAN
IDS = IDS_REDUCTION_MEAN

SUBSAMPLINGS = [None, [0, 0, 1, 0, 1]]
SUBSAMPLINGS_IDS = [f"subsampling={sub}" for sub in SUBSAMPLINGS]

PARAM_GROUPS_FN = PARAM_BLOCKS_FN
PARAM_GROUPS_FN_IDS = PARAM_BLOCKS_FN_IDS


def keep_all(evals: Tensor) -> List[int]:
    """Keep all eigenvalues.

    Args:
        evals: Eigenvalues.

    Returns:
        Indices of eigenvalues to keep.
    """
    return list(range(evals.numel()))


def keep_nonzero(evals: Tensor, min_abs=1e-4) -> List[int]:
    """Keep all eigenvalues that exceed a threshold in magnitude.

    Args:
        evals: Eigenvalues.

    Returns:
        Indices of eigenvalues to keep.
    """
    return [i for i in range(evals.numel()) if evals[i].abs() >= min_abs]
