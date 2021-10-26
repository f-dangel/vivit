"""Test subsampling utilities."""
import pytest
from backpack.extensions import BatchGrad, SqrtGGNExact

from vivit.utils.subsampling import (
    merge_extensions,
    merge_multiple_subsamplings,
    merge_subsamplings,
    sample_output_mapping,
)


def test_sample_output_mapping():
    """Test mapping from samples to output indices."""
    assert sample_output_mapping(None, None) is None

    assert sample_output_mapping([0, 1], None) == [0, 1]
    assert sample_output_mapping([0, 1], [2, 1, 0]) == [2, 1]

    assert sample_output_mapping([0, 0], [2, 1, 0]) == [2, 2]
    assert sample_output_mapping([2, 0], [2, 1, 0]) == [0, 2]

    with pytest.raises(ValueError):
        sample_output_mapping([2, 1, 0], [0, 1])

    with pytest.raises(ValueError):
        sample_output_mapping(None, [0, 1])


def test_merge_subsamplings():
    """Test merging of sub-samplings."""
    assert merge_subsamplings(None, None) is None
    assert merge_subsamplings(None, [0, 1]) is None

    assert merge_subsamplings([0, 1], [2, 3]) == [0, 1, 2, 3]

    assert merge_subsamplings([0, 1], [0, 1]) == [0, 1]

    assert merge_subsamplings([0, 3, 1], [7, 0]) == [0, 1, 3, 7]


def test_multiple_subsamplings():
    """Test merging of multiple sub-samplings."""
    assert merge_multiple_subsamplings([1, 0]) == [0, 1]

    assert merge_multiple_subsamplings([0, 0, 4, 2], [0, 0, 8], [4, 2]) == [0, 2, 4, 8]

    assert merge_multiple_subsamplings([0, 0, 4, 2], None, [4, 2]) is None

    assert merge_multiple_subsamplings(None, [0, 1], None) is None

    with pytest.raises(ValueError):
        merge_multiple_subsamplings()


def test_merge_extensions():
    """Test merging of sub-sampled extensions."""

    assert merge_extensions(
        [(BatchGrad, None), (BatchGrad, [0, 1]), (SqrtGGNExact, [0])]
    ) == {BatchGrad: None, SqrtGGNExact: [0]}

    assert merge_extensions(
        [(BatchGrad, [2, 5, 0, 0]), (BatchGrad, [0, 1]), (SqrtGGNExact, [1, 0])]
    ) == {
        BatchGrad: [0, 1, 2, 5],
        SqrtGGNExact: [0, 1],
    }

    assert merge_extensions(
        [(BatchGrad, [2, 5, 0, 0]), (BatchGrad, [0, 1]), (BatchGrad, [1, 1, 0])]
    ) == {
        BatchGrad: [0, 1, 2, 5],
    }
