"""Test ``lowrank.optim.computations``."""

from test.implementation.optim_autograd import AutogradOptimExtensions
from test.implementation.optim_backpack import BackpackOptimExtensions
from test.optim.settings import IDS_REDUCTION_MEAN, PROBLEMS_REDUCTION_MEAN
from test.utils import check_sizes_and_values

import pytest

from lowrank.utils.subsampling import is_subset

TOP_K = [1, 5]
TOP_K_IDS = [f"top_k={k}" for k in TOP_K]

SUBSAMPLINGS_DIRECTIONS = [
    None,
    [2, 0],
    [0, 0, 1, 0, 1],
]
SUBSAMPLINGS_DIRECTIONS_IDS = [
    f"subsampling_directions={sub}" for sub in SUBSAMPLINGS_DIRECTIONS
]

SUBSAMPLINGS_FIRST = [
    None,
    [1, 0],
    [0, 0, 1, 0, 1],
]
SUBSAMPLINGS_FIRST_IDS = [f"subsampling_first={sub}" for sub in SUBSAMPLINGS_FIRST]

SUBSAMPLINGS_SECOND = [
    None,
    [1, 0],
    [0, 0, 1, 0, 1],
]
SUBSAMPLINGS_SECOND_IDS = [f"subsampling_second={sub}" for sub in SUBSAMPLINGS_SECOND]


@pytest.mark.parametrize(
    "subsampling_directions", SUBSAMPLINGS_DIRECTIONS, ids=SUBSAMPLINGS_DIRECTIONS_IDS
)
@pytest.mark.parametrize(
    "subsampling_first", SUBSAMPLINGS_FIRST, ids=SUBSAMPLINGS_FIRST_IDS
)
@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_computations_gammas_ggn(
    problem, top_k, subsampling_directions, subsampling_first
):
    """Compare optimizer's 1st-order directional derivatives ``γ[n, d]`` along leading
    GGN eigenvectors with autograd.

    Args:
        problem (ExtensionsTestProblem): Test case.
        top_k (int): Number of leading eigenvectors used as directions. Will be clipped
            to ``[1, max]`` with ``max`` the maximum number of nontrivial eigenvalues.
        subsampling_directions ([int] or None): Indices of samples used to compute
            Newton directions. If ``None``, all samples in the batch will be used.
        subsampling_first ([int], optional): Sample indices used for individual
            gradients.
    """
    problem.set_up()

    autograd_res = AutogradOptimExtensions(problem).gammas_ggn(
        top_k,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
    )
    backpack_res = BackpackOptimExtensions(problem).gammas_ggn(
        top_k,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
    )

    # the directions can sometimes point in the opposite direction, leading
    # to gammas of same magnitude but opposite sign.
    autograd_res = autograd_res.abs()
    backpack_res = backpack_res.abs()

    rtol = 5e-3
    atol = 1e-5

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
    problem.tear_down()


@pytest.mark.parametrize(
    "subsampling_directions", SUBSAMPLINGS_DIRECTIONS, ids=SUBSAMPLINGS_DIRECTIONS_IDS
)
@pytest.mark.parametrize(
    "subsampling_second", SUBSAMPLINGS_SECOND, ids=SUBSAMPLINGS_SECOND_IDS
)
@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize("problem", PROBLEMS_REDUCTION_MEAN, ids=IDS_REDUCTION_MEAN)
def test_computations_lambdas_ggn(
    problem, top_k, subsampling_directions, subsampling_second
):
    """Compare optimizer's 2nd-order directional derivatives ``λ[n, d]`` along leading
    GGN eigenvectors with autograd.

    Args:
        problem (ExtensionsTestProblem): Test case.
        top_k (int): Number of leading eigenvectors used as directions. Will be clipped
            to ``[1, max]`` with ``max`` the maximum number of nontrivial eigenvalues.
        subsampling_directions ([int] or None): Indices of samples used to compute
            Newton directions. If ``None``, all samples in the batch will be used.
        subsampling_second ([int], optional): Sample indices used for individual
            curvature matrices.
    """
    problem.set_up()

    # TODO This requires more scalar products be evaluated
    if not is_subset(subsampling_second, subsampling_directions):
        with pytest.raises(NotImplementedError):
            backpack_res = BackpackOptimExtensions(problem).lambdas_ggn(
                top_k,
                subsampling_directions=subsampling_directions,
                subsampling_second=subsampling_second,
            )
        return

    autograd_res = AutogradOptimExtensions(problem).lambdas_ggn(
        top_k,
        subsampling_directions=subsampling_directions,
        subsampling_second=subsampling_second,
    )
    backpack_res = BackpackOptimExtensions(problem).lambdas_ggn(
        top_k,
        subsampling_directions=subsampling_directions,
        subsampling_second=subsampling_second,
    )

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
