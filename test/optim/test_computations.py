"""Test ``lowrank.optim.computations``."""

from test.implementation.optim_autograd import AutogradOptimExtensions
from test.implementation.optim_backpack import BackpackOptimExtensions
from test.optim.settings import IDS_REDUCTION_MEAN, PROBLEMS_REDUCTION_MEAN
from test.utils import check_sizes_and_values

import pytest

from lowrank.optim.damping import ConstantDamping

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

CONSTANT_DAMPING_VALUES = [1.0]
DAMPINGS = [ConstantDamping(const) for const in CONSTANT_DAMPING_VALUES]
DAMPINGS_IDS = [
    f"damping=ConstantDamping({const})" for const in CONSTANT_DAMPING_VALUES
]


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

    rtol, atol = 1e-5, 1e-6
    check_sizes_and_values(autograd_res, backpack_res, rtol=rtol, atol=atol)
    problem.tear_down()


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
):
    """Compare damped Newton step along leading GGN eigenvectors with autograd.

    Args:
        problem (ExtensionsTestProblem): Test case.
        top_k (int): Number of leading eigenvectors used as directions. Will be clipped
            to ``[1, max]`` with ``max`` the maximum number of nontrivial eigenvalues.
        damping (lowrank.optim.damping.BaseDamping): Policy for selecting dampings along
            a direction from first- and second- order directional derivatives.
        subsampling_directions ([int] or None): Indices of samples used to compute
            Newton directions. If ``None``, all samples in the batch will be used.
        subsampling_first ([int], optional): Sample indices used for individual
            gradients.
        subsampling_second ([int], optional): Sample indices used for individual
            curvature matrices.
    """
    problem.set_up()

    autograd_res = AutogradOptimExtensions(problem).newton_step(
        top_k,
        damping,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
        subsampling_second=subsampling_second,
    )
    backpack_res = BackpackOptimExtensions(problem).newton_step(
        top_k,
        damping,
        subsampling_directions=subsampling_directions,
        subsampling_first=subsampling_first,
        subsampling_second=subsampling_second,
    )

    atol = 5e-6
    rtol = 5e-4

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
    problem.tear_down()
