"""Tests computations of first-order directional derivatives ``vivit.optim``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.optim.settings import (
    IDS_REDUCTION_MEAN,
    PARAM_BLOCKS_FN,
    PARAM_BLOCKS_FN_IDS,
    PROBLEMS_REDUCTION_MEAN,
    SUBSAMPLINGS_DIRECTIONS,
    SUBSAMPLINGS_DIRECTIONS_IDS,
    SUBSAMPLINGS_FIRST,
    SUBSAMPLINGS_FIRST_IDS,
    TOP_K,
    TOP_K_IDS,
    insert_criterion,
)
from test.utils import check_sizes_and_values

import pytest


@pytest.mark.expensive
@pytest.mark.parametrize("param_block_fn", PARAM_BLOCKS_FN, ids=PARAM_BLOCKS_FN_IDS)
@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize(
    "subsampling_first", SUBSAMPLINGS_FIRST, ids=SUBSAMPLINGS_FIRST_IDS
)
@pytest.mark.parametrize(
    "subsampling_directions", SUBSAMPLINGS_DIRECTIONS, ids=SUBSAMPLINGS_DIRECTIONS_IDS
)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_gamma_ggn(
    problem, top_k, subsampling_directions, subsampling_first, param_block_fn
):
    """Compare first-order directional derivatives ``γ[n, d]`` with autograd.

    Args:
        top_k (function): Criterion to select Gram space directions.
        subsampling_directions ([int], optional): Sample indices used for the GGN.
        subsampling_first ([int], optional): Sample indices used for individual
            gradients.
        param_block_fn (function): Function to group model parameters.
    """
    problem.set_up()

    param_groups = param_block_fn(problem.model.parameters())
    insert_criterion(param_groups, top_k)

    autograd_res = AutogradExtensions(problem).gammas_ggn(
        param_groups,
        ggn_subsampling=subsampling_directions,
        grad_subsampling=subsampling_first,
    )
    backpack_res = BackpackExtensions(problem).gammas_ggn(
        param_groups,
        ggn_subsampling=subsampling_directions,
        grad_subsampling=subsampling_first,
    )

    # directions can vary in sign, leading to same magnitude but opposite sign.
    autograd_res = [res.abs() for res in autograd_res]
    backpack_res = [res.abs() for res in backpack_res]

    rtol = 5e-3
    atol = 1e-4

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
    problem.tear_down()
