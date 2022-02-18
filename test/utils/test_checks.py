"""Test ``vivit.utils.checks``."""

from pytest import raises
from torch import rand

from vivit.utils.checks import (
    check_key_exists,
    check_subsampling_unique,
    check_unique_params,
)


def test_missing_key():
    """Test detection of missing 'params' in parameter groups."""
    param_groups = [{"param": []}]

    with raises(ValueError):
        check_key_exists(param_groups, "params")


def test_unique_params():
    """Test detection of parameters assigned to multiple parameter groups."""
    p1, p2 = rand(10), rand(5)
    param_groups = [
        {"params": [p1, p2]},
        {"params": [p1]},
    ]

    with raises(ValueError):
        check_unique_params(param_groups)


def test_subsampling_unique():
    """Test detection of dduplicate sub-sampling inddices."""
    with raises(ValueError):
        check_subsampling_unique([0, 0, 1])
