"""Test ``vivit.optim.damped_newton``."""

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
from test.problem import ExtensionsTestProblem
from test.utils import check_sizes_and_values
from typing import Any, Callable, Dict, Iterator, List, Union

import pytest
from torch import Tensor

from vivit.optim.damping import ConstantDamping, _DirectionalCoefficients

CONSTANT_DAMPING_VALUES = [1.0]
DAMPINGS = [ConstantDamping(const) for const in CONSTANT_DAMPING_VALUES]
DAMPINGS_IDS = [
    f"damping=ConstantDamping({const})" for const in CONSTANT_DAMPING_VALUES
]

USE_CLOSURE = [False, True]
USE_CLOSURE_IDS = [f"use_closure={use}" for use in USE_CLOSURE]


@pytest.mark.parametrize("use_closure", USE_CLOSURE, ids=USE_CLOSURE_IDS)
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
def test_optim_newton_step(
    problem: ExtensionsTestProblem,
    top_k: Callable[[Tensor], List[int]],
    damping: _DirectionalCoefficients,
    subsampling_directions: Union[List[int], None],
    subsampling_first: Union[List[int], None],
    subsampling_second: Union[List[int], None],
    param_block_fn: Callable[[Iterator[Tensor]], List[Dict[str, Any]]],
    use_closure: bool,
):
    """Compare damped Newton step along leading GGN eigenvectors with autograd.

    Use ``DampedNewton`` optimizer to compute Newton steps.

    Args:
        top_k: Criterion to select Gram space directions.
        problem: Test case.
        damping: Policy for selecting dampings along
            a direction from first- and second- order directional derivatives.
        subsampling_directions: Indices of samples used to compute
            Newton directions. If ``None``, all samples in the batch will be used.
        subsampling_first: Sample indices used for individual gradients.
        subsampling_second: Sample indices used for individual curvature matrices.
        param_block_fn: Function to group model parameters.
        use_closure: Whether to use a closure for computing the Newton step.
    """
    problem.set_up()

    param_groups = param_block_fn(problem.model.named_parameters())
    insert_criterion(param_groups, top_k)

    autograd_res = AutogradOptimExtensions(problem).newton_step(
        param_groups,
        damping,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
        subsampling_second=subsampling_second,
    )
    backpack_res = BackpackOptimExtensions(problem).optim_newton_step(
        param_groups,
        damping,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
        subsampling_second=subsampling_second,
        use_closure=use_closure,
    )

    atol = 5e-5
    rtol = 1e-4

    assert len(autograd_res) == len(backpack_res) == len(param_groups)
    for autograd_step, backpack_step in zip(autograd_res, backpack_res):
        check_sizes_and_values(autograd_step, backpack_step, atol=atol, rtol=rtol)

    problem.tear_down()
