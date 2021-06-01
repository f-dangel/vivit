"""Test ``vivit.optim.gram_computations``."""

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


@pytest.mark.parametrize("param_block_fn", PARAM_BLOCKS_FN, ids=PARAM_BLOCKS_FN_IDS)
@pytest.mark.parametrize(
    "subsampling_directions", SUBSAMPLINGS_DIRECTIONS, ids=SUBSAMPLINGS_DIRECTIONS_IDS
)
@pytest.mark.parametrize(
    "subsampling_first", SUBSAMPLINGS_FIRST, ids=SUBSAMPLINGS_FIRST_IDS
)
@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_computations_gammas_ggn(
    problem, top_k, subsampling_directions, subsampling_first, param_block_fn
):
    """Compare optimizer's 1st-order directional derivatives ``γ[n, d]`` along leading
    GGN eigenvectors with autograd.

    Args:
        problem (ExtensionsTestProblem): Test case.
        top_k (function): Criterion to select Gram space directions.
        subsampling_directions ([int] or None): Indices of samples used to compute
            Newton directions. If ``None``, all samples in the batch will be used.
        subsampling_first ([int], optional): Sample indices used for individual
            gradients.
        param_block_fn (function): Function to group model parameters.
    """
    problem.set_up()

    param_groups = param_block_fn(problem.model.parameters())
    insert_criterion(param_groups, top_k)

    autograd_res = AutogradOptimExtensions(problem).gammas_ggn(
        param_groups,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
    )
    backpack_res = BackpackOptimExtensions(problem).gammas_ggn(
        param_groups,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
    )

    # directions can vary in sign, leading to same magnitude but opposite sign.
    autograd_res = [res.abs() for res in autograd_res]
    backpack_res = [res.abs() for res in backpack_res]

    rtol = 5e-3
    atol = 1e-4

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
    problem.tear_down()


@pytest.mark.parametrize("param_block_fn", PARAM_BLOCKS_FN, ids=PARAM_BLOCKS_FN_IDS)
@pytest.mark.parametrize(
    "subsampling_directions", SUBSAMPLINGS_DIRECTIONS, ids=SUBSAMPLINGS_DIRECTIONS_IDS
)
@pytest.mark.parametrize(
    "subsampling_second", SUBSAMPLINGS_SECOND, ids=SUBSAMPLINGS_SECOND_IDS
)
@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_computations_lambdas_ggn(
    problem, top_k, subsampling_directions, subsampling_second, param_block_fn
):
    """Compare optimizer's 2nd-order directional derivatives ``λ[n, d]`` along leading
    GGN eigenvectors with autograd.

    Args:
        problem (ExtensionsTestProblem): Test case.
        top_k (function): Criterion to select Gram space directions.
        subsampling_directions ([int] or None): Indices of samples used to compute
            Newton directions. If ``None``, all samples in the batch will be used.
        subsampling_second ([int], optional): Sample indices used for individual
            curvature matrices.
        param_block_fn (function): Function to group model parameters.
    """
    problem.set_up()

    param_groups = param_block_fn(problem.model.parameters())
    insert_criterion(param_groups, top_k)

    autograd_res = AutogradOptimExtensions(problem).lambdas_ggn(
        param_groups,
        subsampling_directions=subsampling_directions,
        subsampling_second=subsampling_second,
    )
    backpack_res = BackpackOptimExtensions(problem).lambdas_ggn(
        param_groups,
        subsampling_directions=subsampling_directions,
        subsampling_second=subsampling_second,
    )

    rtol, atol = 1e-5, 1e-5
    check_sizes_and_values(autograd_res, backpack_res, rtol=rtol, atol=atol)
    problem.tear_down()
