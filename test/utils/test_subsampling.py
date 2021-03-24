"""Test subsampling utilities."""
from lowrank.utils.subsampling import merge_subsamplings, sample_output_mapping


def test_sample_output_mapping():
    """Test mapping from samples to output indices."""
    assert sample_output_mapping([0, 1], None) == [0, 1]
    assert sample_output_mapping([0, 1], [2, 1, 0]) == [2, 1]

    assert sample_output_mapping([0, 0], [2, 1, 0]) == [2, 2]
    assert sample_output_mapping([2, 0], [2, 1, 0]) == [0, 2]


def test_merge_subsamplings():
    """Test merging of sub-samplings."""
    assert merge_subsamplings(None, None) is None
    assert merge_subsamplings(None, [0, 1]) is None

    assert merge_subsamplings([0, 1], [2, 3]) == [0, 1, 2, 3]

    assert merge_subsamplings([0, 1], [0, 1]) == [0, 1]

    assert merge_subsamplings([0, 3, 1], [7, 0]) == [0, 1, 3, 7]
