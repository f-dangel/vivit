"""Test ``vivit.optim.computations``."""

from test.implementation.optim_autograd import AutogradOptimExtensions
from test.implementation.optim_backpack import BackpackOptimExtensions
from test.optim.settings import (
    IDS_REDUCTION_MEAN,
    PARAM_BLOCKS_FN,
    PARAM_BLOCKS_FN_IDS,
    PROBLEMS_REDUCTION_MEAN,
    SUBSAMPLINGS_DIRECTIONS,
    SUBSAMPLINGS_DIRECTIONS_IDS,
    SUBSAMPLINGS_FIRST,
    SUBSAMPLINGS_FIRST_IDS,
    SUBSAMPLINGS_SECOND,
    SUBSAMPLINGS_SECOND_IDS,
    TOP_K,
    TOP_K_IDS,
    insert_criterion,
)
from test.utils import check_sizes_and_values

import pytest

from vivit.optim.damping import ConstantDamping

CONSTANT_DAMPING_VALUES = [1.0]
DAMPINGS = [ConstantDamping(const) for const in CONSTANT_DAMPING_VALUES]
DAMPINGS_IDS = [
    f"damping=ConstantDamping({const})" for const in CONSTANT_DAMPING_VALUES
]


@pytest.mark.parametrize("param_block_fn", PARAM_BLOCKS_FN, ids=PARAM_BLOCKS_FN_IDS)
@pytest.mark.parametrize("damping", DAMPINGS, ids=DAMPINGS_IDS)
@pytest.mark.parametrize(
    "subsampling_directions", SUBSAMPLINGS_DIRECTIONS, ids=SUBSAMPLINGS_DIRECTIONS_IDS
)
@pytest.mark.parametrize(
    "subsampling_first", SUBSAMPLINGS_FIRST, ids=SUBSAMPLINGS_FIRST_IDS
)
@pytest.mark.parametrize(
    "subsampling_second", SUBSAMPLINGS_SECOND, ids=SUBSAMPLINGS_SECOND_IDS
)
@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_computations_newton_step(
    problem,
    top_k,
    damping,
    subsampling_directions,
    subsampling_first,
    subsampling_second,
    param_block_fn,
):
    """Compare damped Newton step along leading GGN eigenvectors with autograd.

    Args:
        top_k (function): Criterion to select Gram space directions.
        problem (ExtensionsTestProblem): Test case.
        damping (vivit.optim.damping.BaseDamping): Policy for selecting dampings along
            a direction from first- and second- order directional derivatives.
        subsampling_directions ([int] or None): Indices of samples used to compute
            Newton directions. If ``None``, all samples in the batch will be used.
        subsampling_first ([int], optional): Sample indices used for individual
            gradients.
        subsampling_second ([int], optional): Sample indices used for individual
            curvature matrices.
        param_block_fn (function): Function to group model parameters.
    """
    problem.set_up()

    param_groups = param_block_fn(problem.model.parameters())
    insert_criterion(param_groups, top_k)

    autograd_res = AutogradOptimExtensions(problem).newton_step(
        param_groups,
        damping,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
        subsampling_second=subsampling_second,
    )
    backpack_res = BackpackOptimExtensions(problem).newton_step(
        param_groups,
        damping,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
        subsampling_second=subsampling_second,
    )

    atol = 5e-5
    rtol = 1e-4

    assert len(autograd_res) == len(backpack_res) == len(param_groups)
    for autograd_step, backpack_step in zip(autograd_res, backpack_res):
        check_sizes_and_values(autograd_step, backpack_step, atol=atol, rtol=rtol)

    problem.tear_down()
